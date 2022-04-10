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

import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import *
import numpy as np
from tqdm import tqdm
import os
import argparse

import models.vgg as models_vgg


parser = argparse.ArgumentParser(description='PyTorch FoolBox Evaluation')
parser.add_argument('--arch', help='backbone model arch')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--attack', help='attack type')
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
    args.gpu = gpu
    attack = args.attack
    os.makedirs(args.output, exist_ok=True)
    acc_path = os.path.join(args.output, f'{args.attack}.txt')

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
    if 'imagenet_100' in args.data:
        num_classes = 100
    else:
        num_classes = 1000
    
    if args.rank == 0:
        print("%s-way classification" % num_classes)

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model_name = str(args.arch)
        model = models_vgg.__dict__[args.arch](pretrained=True, num_classes=num_classes)
    else:
        print("=> creating model '{}'".format(args.arch))
        model_name = str(args.arch)
        model = models_vgg.__dict__[model_name](pretrained=False, num_classes=num_classes)

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
        else:
            checkpoint['model'][value] = checkpoint_old['model'][value]

    model.load_state_dict(checkpoint['model'], strict=False)
    print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint, checkpoint_old['epoch']))
    model.eval()

    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(160),
            transforms.ToTensor(),])), batch_size=1, shuffle=False)

    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing, device="cuda")

    # apply the attack
    attack = globals()[attack]()
    epsilons = [
        0.0000,
        0.0002,
        0.0004,
        0.0006,
        0.0008,
        0.0010,
        0.0012,
        0.0014,
        0.0016,
        0.0018,
        0.0020,
    ]

    clean_acc, robust_acc = validate(fmodel=fmodel, val_loader=val_loader, attack=attack, epsilons=epsilons)

    f = open(acc_path,'a')
    print(f"clean accuracy:  {clean_acc} %", file=f)
    print("robust accuracy for perturbations with", file=f)
    for eps, acc in zip(epsilons, robust_acc):
        print(f" {args.attack} norm ≤ {eps:<6}: {acc * 100} %", file=f)
    f.close()

def validate(fmodel, val_loader, attack, epsilons):
    clean_acc_list = list()
    robust_acc_list = list()

    for i, (images, labels) in enumerate(tqdm(val_loader)):
        images = images.cuda()
        images = ep.PyTorchTensor(images)
        labels = labels.cuda()
        labels = ep.PyTorchTensor(labels)

        accuracy(fmodel, images, labels)

        clean_acc = accuracy(fmodel, images, labels)
        clean_acc_list.append(clean_acc * 100)
        print(clean_acc * 100)

        raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons)
        robust_accuracy = 1 - success.float32().mean(axis=-1)
        robust_acc_list.append(robust_accuracy.raw)

    return np.mean(clean_acc_list), torch.stack(robust_acc_list).mean(dim=0).tolist()

if __name__ == "__main__":
    main()