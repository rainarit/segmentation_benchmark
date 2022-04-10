#!/usr/bin/env python3
"""
A simple example that demonstrates how to run a single attack against
a PyTorch ResNet-18 model for different epsilons and how to then report
the robust accuracy.
"""
from turtle import distance
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.distributed as dist
import torch
import torch.multiprocessing as mp
import warnings
from enum import Enum
import csv

import numpy as np
from tqdm import tqdm
import os
import argparse

import models.vgg as models_vgg


parser = argparse.ArgumentParser(description='PyTorch FoolBox Evaluation')
parser.add_argument('--arch', help='backbone model arch')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--pertubation', help='pertubation type')
parser.add_argument('--num-classes', help='number of classes', type=int)
parser.add_argument('--output', help='name of output dir')
parser.add_argument('--checkpoint', help='name of checkpoint path')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1024, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--inplanes', default=64, type=int,
                    help='Number of inplanes (conv1 - ResNet)')
parser.add_argument('--tensorboard-dir', default='runs', help='path where to save tensorboard')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

def main():
    args = parser.parse_args()

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    print(args.data)
    args.gpu = gpu
    os.makedirs(args.output, exist_ok=True)
    acc_path = os.path.join(args.output, f'{args.pertubation}.csv')

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model_name = str(args.arch)
        model = models_vgg.__dict__[args.arch](pretrained=True, num_classes=args.num_classes)
    else:
        print("=> creating model '{}'".format(args.arch))
        model_name = str(args.arch)
        model = models_vgg.__dict__[model_name](pretrained=False, num_classes=args.num_classes)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    
    print("=> loading checkpoint '{}'".format(args.checkpoint))
    checkpoint = {'model': {}}
    if args.gpu is None:
        checkpoint_old = torch.load(args.checkpoint)
    else:
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint_old = torch.load(args.checkpoint, map_location=loc)

    for value in checkpoint_old['model']:
        if 'module' not in value:
            name = 'module.' + value
            checkpoint['model'][name] = checkpoint_old['model'][value]

    model.load_state_dict(checkpoint['model'], strict=True)
    print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint, checkpoint_old['epoch']))
    model.eval()

    val_dataset = datasets.ImageFolder(args.data, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),]))

    val_dataset_100_class_to_idx = {'n01558993': 0, 'n01692333': 1, 'n01729322': 2, 'n01735189': 3, 'n01749939': 4, 
                                    'n01773797': 5, 'n01820546': 6, 'n01855672': 7, 'n01978455': 8, 'n01980166': 9, 
                                    'n01983481': 10, 'n02009229': 11, 'n02018207': 12, 'n02085620': 13, 'n02086240': 14, 
                                    'n02086910': 15, 'n02087046': 16, 'n02089867': 17, 'n02089973': 18, 'n02090622': 19, 
                                    'n02091831': 20, 'n02093428': 21, 'n02099849': 22, 'n02100583': 23, 'n02104029': 24, 
                                    'n02105505': 25, 'n02106550': 26, 'n02107142': 27, 'n02108089': 28, 'n02109047': 29, 
                                    'n02113799': 30, 'n02113978': 31, 'n02114855': 32, 'n02116738': 33, 'n02119022': 34, 
                                    'n02123045': 35, 'n02138441': 36, 'n02172182': 37, 'n02231487': 38, 'n02259212': 39, 
                                    'n02326432': 40, 'n02396427': 41, 'n02483362': 42, 'n02488291': 43, 'n02701002': 44, 
                                    'n02788148': 45, 'n02804414': 46, 'n02859443': 47, 'n02869837': 48, 'n02877765': 49, 
                                    'n02974003': 50, 'n03017168': 51, 'n03032252': 52, 'n03062245': 53, 'n03085013': 54, 
                                    'n03259280': 55, 'n03379051': 56, 'n03424325': 57, 'n03492542': 58, 'n03494278': 59, 
                                    'n03530642': 60, 'n03584829': 61, 'n03594734': 62, 'n03637318': 63, 'n03642806': 64, 
                                    'n03764736': 65, 'n03775546': 66, 'n03777754': 67, 'n03785016': 68, 'n03787032': 69, 
                                    'n03794056': 70, 'n03837869': 71, 'n03891251': 72, 'n03903868': 73, 'n03930630': 74, 
                                    'n03947888': 75, 'n04026417': 76, 'n04067472': 77, 'n04099969': 78, 'n04111531': 79, 
                                    'n04127249': 80, 'n04136333': 81, 'n04229816': 82, 'n04238763': 83, 'n04336792': 84, 
                                    'n04418357': 85, 'n04429376': 86, 'n04435653': 87, 'n04485082': 88, 'n04493381': 89, 
                                    'n04517823': 90, 'n04589890': 91, 'n04592741': 92, 'n07714571': 93, 'n07715103': 94, 
                                    'n07753275': 95, 'n07831146': 96, 'n07836838': 97, 'n13037406': 98, 'n13040303': 99}

    val_dataset.samples = []
    for cls, index in val_dataset_100_class_to_idx.items():
        dir_path = str(args.data)+str(cls)
        for filename in os.listdir(dir_path):
            f = os.path.join(dir_path, filename)
            # checking if it is a file
            if os.path.isfile(f):
                val_dataset.samples.append((str(f), index))
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, num_workers= args.workers, shuffle=False, pin_memory=True)

    top1_avg, top5_avg = validate(val_loader, model, args)
    with open(acc_path, "w") as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow([float(top1_avg), float(top5_avg)])
    

def validate(val_loader, model, args):
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [top1, top5],
        prefix='Test: ')

    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(val_loader)):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # Casts operations to mixed precision
            with torch.cuda.amp.autocast():
                output = model(images)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

    return top1.avg, top5.avg

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
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
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

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

if __name__ == "__main__":
    main()