import argparse
import os
import random
import shutil
import sys
import string
import time
import warnings
import math
import numpy as np
import json
from PIL import Image

import torch
from pathlib import Path
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms.functional as TF

from torch.utils.tensorboard import SummaryWriter

import models.vgg as models_vgg
import utils



model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18')
parser.add_argument('-b', '--batch-size', default=1024, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-ol', '--occlude_low', default=0., type=float,
                    help='lower bound of scale range for occlusion perturbation')
parser.add_argument('-oh', '--occlude_high', default=0.001, type=float,
                    help='upper bound of scale range for occlusion perturbation')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--output', help='name of output dir')

best_acc1 = 0
global_stats_file = None

"""NOTE:
If you are using NCCL backend, remember to set 
the following environment variables in your Ampere series GPU.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
"""

seed = 56
torch.manual_seed(seed)
np.random.seed(seed)


def main():
    args = parser.parse_args()
    if args.seed is not None:
        # random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main_worker(args.gpu, args)

def generate_rand_string(n):
  letters = string.ascii_lowercase
  str_rand = ''.join(random.choice(letters) for i in range(n))
  return str_rand

def main_worker(gpu, args):
    args.checkpoint_dir = Path('/'.join(args.resume.split('/')[:-1]))
    while True:
        output_dir = Path("%s/output_%s_imagenet100_occlude_%s_%s_%s" % (args.output, args.arch,
                                                                         args.occlude_low, args.occlude_high, 
                                                                         generate_rand_string(6)))
        if not os.path.exists(output_dir):
            args.output = output_dir
            break
    
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    if 'imagenet_100' in args.data:
        num_classes = 100
    else:
        num_classes = 1000
    
    print("=> creating model '{}'".format(args.arch))
    model_name = str(args.arch)
    model = models_vgg.__dict__[args.arch](pretrained=False, num_classes=num_classes)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        print("Set to GPU", args.gpu)
    
    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optionally resume from a checkpoint
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        if args.gpu is None:
            checkpoint = torch.load(args.resume)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
        args.start_epoch = checkpoint['epoch']
        print(checkpoint.keys())
        best_acc1 = checkpoint['val_acc1']
        if 'dale' in args.arch:
            new_checkpoint = {}
            for key in checkpoint['model'].keys():
                new_key = key
                if 'module' in key:
                    new_key = key.split('module.')[1]
                new_checkpoint[new_key] = checkpoint['model'][key]
            model.load_state_dict(new_checkpoint)
        else:
            model.load_state_dict(checkpoint['model'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    valdir = os.path.join(args.data, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(160),
            transforms.ToTensor(),
            Occlude(args.occlude_low, args.occlude_high, 0.),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=1, pin_memory=True)
    
    val_acc1, val_acc5 = validate(val_loader, model, criterion, args)
    # remember best acc@1 and save checkpoint
    occlude_dict = {"occlude_low": args.occlude_low, "occlude_high": args.occlude_high, "val_accuracy1": val_acc1.item(), "val_accuracy5": val_acc5.item()}
    with open(args.output / "results.csv", 'w') as f:
        for key in occlude_dict.keys():
            f.write("%s,%s\n"%(key,occlude_dict[key]))
    #print(json.dumps(occlude_dict), file=global_stats_file)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class Occlude(nn.Module):
  """
  Apply occlusion to a patch of pixels in images.
  Due to center-bias present in natural images from ImageNet, a 2-D Gaussian is 
  used to sample the location of the patch.
  """
  def __init__(self, scale_low, scale_high, value):
    super().__init__()
    self.scale = (scale_low, scale_high)
    self.value = value

  @staticmethod
  def get_params(img, scale, value=0.):
    img_h, img_w = img.shape[1], img.shape[2]
    image_area = img_h * img_w
    erase_area = image_area * torch.empty(1).uniform_(scale[0], scale[1]).item()
    for i in range(10):
      erase_h = int(np.sqrt(erase_area))
      erase_w = int(np.sqrt(erase_area))
      if not (erase_h < img_h and erase_w < img_w):
        continue
      
      i = np.random.randint(0, img_h - erase_h + 1)
      j = np.random.randint(0, img_w - erase_w + 1)
      return i, j, erase_h, erase_w, value
    return 0, 0, 
    
  def forward(self, img):
    if isinstance(img, Image.Image):
      img = F.to_tensor(img)
    i, j, erase_h, erase_w, value = self.get_params(img=img, scale=self.scale, value=self.value)
    return TF.erase(img, i, j, erase_h, erase_w, value, True)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()