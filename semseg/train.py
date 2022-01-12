import datetime
import os
import time
import torch
import torch.utils.data
import transforms as T
from torch import nn
import torchvision
import numpy as np
import scipy.io
import random
from PIL import Image
import matplotlib.image as mpimg
from tqdm import tqdm
from coco_utils import get_coco
import presets
import utils
from torchvision import utils as torch_utils
import os
import sys
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import math

from torchvision.utils import save_image
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from models.segmentation.segmentation import _load_model as load_model
import  csv
from torchvision.utils import save_image


def get_dataset(dir_path, name, image_set, transform):
    def sbd(*args, **kwargs):
        return torchvision.datasets.SBDataset(*args, mode='segmentation', **kwargs)
    paths = {
        "coco": (dir_path, get_coco, 21), 
        "voc": (dir_path, torchvision.datasets.VOCSegmentation, 21),
        "voc_aug": (dir_path, sbd, 21)
    }
    p, ds_fn, num_classes = paths[name]
    if name == "voc":
        ds = ds_fn(p, year="2012", image_set=image_set, transforms=transform, download=False)
    elif name == "voc_aug":
        ds = ds_fn(p, image_set=image_set, transforms=transform, download=False)
    else:
        ds = ds_fn(p, image_set=image_set, transforms=transform, download=False)
    return ds, num_classes

def get_transform(train):
    base_size = 520
    crop_size = 480
    return presets.SegmentationPresetTrain(base_size, crop_size) if train else presets.SegmentationPresetEval(base_size, contrast=args.contrast)

def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255, reduction='mean')
    if len(losses) == 1:
        return losses['out']
    return losses['out'] + 0.5 * losses['aux']

def evaluate(model, data_loader, device, num_classes, output_dir):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    confmat = utils.ConfusionMatrix(num_classes)
    per_mean_iou = list()

    image_epoch_dir = os.path.join(output_dir, "images/")
    utils.mkdir(image_epoch_dir)
    image_dir = os.path.join(output_dir, "images/image/")
    utils.mkdir(image_dir)
    target_dir = os.path.join(output_dir, "images/target/")
    utils.mkdir(target_dir)
    prediction_dir = os.path.join(output_dir, "images/prediction/")
    utils.mkdir(prediction_dir)
    mean_iou_file = os.path.join(output_dir, "per_image_mean_iou.csv")

    with torch.no_grad():
        start_time = time.time()
        for idx, (image, target) in enumerate(metric_logger.log_every(data_loader, 1, header)):
            image, target = image.to(device), target.to(device)

            confmat_image = utils.ConfusionMatrix(num_classes)

            output = model(image)
            output = output['out']

            inv_normalize = T.Normalize(mean=(-0.485, -0.456, -0.406), std=(1/0.229, 1/0.224, 1/0.225))
            
            image_path =  os.path.join(image_dir, '{}.npy'.format(idx))
            target_path = os.path.join(target_dir, '{}.npy'.format(idx))
            prediction_path = os.path.join(prediction_dir, '{}.npy'.format(idx))

            utils.save_on_master(inv_normalize(image[0], target)[0], image_path)
            utils.save_on_master(target, target_path)
            utils.save_on_master(output, prediction_path)

            confmat.update(target.flatten(), output.argmax(1).flatten())

            confmat_image.update(target.flatten(), output.argmax(1).flatten())
            acc_global, acc, iu = confmat_image.compute()
            confmat_image.reduce_from_all_processes()

            image_mean_iou = list((iu * 100).tolist())
            per_mean_iou.append(image_mean_iou)

        confmat.reduce_from_all_processes()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Validation time {}'.format(total_time_str))
    
    with open(mean_iou_file, 'w', newline='') as f:
            wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            wr.writerow(per_mean_iou)

    return confmat

def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq):

    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ") 
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)

    # Creates once at the beginning of training
    scaler = torch.cuda.amp.GradScaler()

    for idx, (image, target) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, target = image.to(device), target.to(device)

        # Casts operations to mixed precision
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(image)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        lr_scheduler.step()

        # Clamping parameters of divnorm to non-negative values
        if "divnorm" in str(args.backbone):
            for module_name, module in model.named_modules():
                if module_name.endswith("div"):
                    curr_module_weight = module.weight.data
                    curr_module_weight = curr_module_weight.clamp(min=0.)
                    module.weight.data = curr_module_weight

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

def main(args):

    seed=args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(seed)

    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    dataset, num_classes = get_dataset(args.data_path, args.dataset, "train", get_transform(train=True))

    dataset_test, _ = get_dataset(args.data_path, args.dataset, "val", get_transform(train=False))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler, 
        num_workers=args.workers, 
        collate_fn=utils.collate_fn, 
        drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, 
        batch_size=1, 
        sampler=test_sampler, 
        num_workers=args.workers, 
        collate_fn=utils.collate_fn)


    model = load_model(arch_type=args.model, 
                       backbone=args.backbone,
                       pretrained=False,
                       progress=True, 
                       num_classes=num_classes, 
                       aux_loss=args.aux_loss, 
                       divnorm_fsize=5)
    
    model.to(device)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    params_to_optimize = [
        {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
    ]

    if args.aux_loss:
        params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})
    
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)
    
    if args.test_only:

        iou_file = os.path.join('/home/AD/rraina/segmentation_benchmark/semseg/csv/', str(args.output) + "_test.csv")
        iou_image_file = os.path.join('/home/AD/rraina/segmentation_benchmark/semseg/csv/', str(args.output) + "per_image_mean_test.csv")

        if args.resume != '':
            checkpoint = torch.load(str(args.resume))
            model_without_ddp.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        
        confmat, per_image_mean = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
        confmat_iu = confmat.get_IoU()

        writer=csv.writer(open(iou_file,'w'))
        print(confmat_iu)
        writer.writerow([confmat_iu])

        writer=csv.writer(open(iou_image_file,'w'))
        for image_mean_iou in per_image_mean:
            writer.writerow([image_mean_iou])

        print(confmat)
        return
    
    output_dir = os.path.join('/home/AD/rraina/segmentation_benchmark/semseg/output/', args.output)
    output_val_dir = os.path.join(output_dir, "val/")
    output_checkpoints_dir = os.path.join(output_dir, "checkpoints/")
    output_val_epochs_dir = os.path.join(output_dir, "val/epochs/")
    if not(os.path.isdir(output_dir)):
        utils.mkdir(output_dir)
    if not(os.path.isdir(output_val_dir)):
        utils.mkdir(output_val_dir)
    if not(os.path.isdir(output_checkpoints_dir)):
        utils.mkdir(output_checkpoints_dir)
    if not(os.path.isdir(output_val_epochs_dir)):
        utils.mkdir(output_val_epochs_dir)
    
    mean_iou_file = os.path.join(output_dir, "mean_iou.csv")
    mean_iou_list = list()

    per_image_mean_iou_file = str(args.output) + "_per_image_mean_iou.csv"

    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, args.print_freq)

        epoch_dir = os.path.join(output_val_epochs_dir, "epoch_{}/".format(epoch))
        utils.mkdir(epoch_dir)

        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes, output_dir=epoch_dir)
        print(confmat)

        confmat_iu = confmat.get_IoU()
        confmat_acc_global = confmat.get_acc_global_correct()
        mean_iou_list.append(confmat_iu)

        checkpoint = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args
        }
        utils.save_on_master(checkpoint, os.path.join(output_checkpoints_dir, 'checkpoint_{}.pth'.format(epoch)))

    with open(mean_iou_file, 'w', newline='') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(mean_iou_list)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Training', add_help=add_help)

    parser.add_argument('--data-path', default='/home/AD/rraina/segmentation_benchmark/', help='dataset path')
    parser.add_argument('--seed', default=429, type=float, help='seed')
    parser.add_argument('--dataset', default='coco', help='dataset name')
    parser.add_argument('--model', default='deeplabv3', help='model')
    parser.add_argument('--backbone', default='resnet101', help='backbone')
    parser.add_argument('--aux-loss', action='store_true', help='auxiliar loss')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--contrast', default=1.0, type=float)
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output', default='./deeplabv3resnet50', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
