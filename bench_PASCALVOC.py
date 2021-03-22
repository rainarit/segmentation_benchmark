from __future__ import print_function
import torchvision
from collections import defaultdict, deque
import math
import time
import torch.distributed as dist
import errno
import os
import numpy as np
from PIL import Image
import random
import datetime
import torch
import torch.utils.data
import transforms as T
import argparse
from tqdm import tqdm

from MetricLogger import MetricLogger
from ConfusionMatrix import ConfusionMatrix
from SmoothedValue import SmoothedValue

from evaluate import get_transform, get_transform_eval, get_transform_train, collate_fn, evaluate

from pytorch_boundaries.models.recurrence.v1net import ReducedV1Net, V1Net

def main(args):
    MODEL_NAME = args.model
    PRETRAINED = args.pretrained
    DEVICE = args.device
    BATCH_SIZE = args.batch_size
    WORKERS = args.workers

    print("Selected Model:", MODEL_NAME)
    if PRETRAINED == True:
        print("Selected Pre-trained = True")
    else:
        print("Selected Pre-trained = False")
    print("------------------------------------------------------------------------------------")
    model = torchvision.models.segmentation.__dict__[MODEL_NAME](num_classes=21, 
                                                                 pretrained=PRETRAINED)
    print("Downloaded", MODEL_NAME, "successfully!")
    print("------------------------------------------------------------------------------------")

    print("Downloading PASCAL VOC 2012 Validation Set")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # Downloading PASCAL VOC 2012 Validation Set
    try:
        dataset_val = torchvision.datasets.VOCSegmentation(root=str(dir_path), 
                                                           year='2012', 
                                                           image_set="val", 
                                                           transforms=get_transform(train=False), 
                                                           download=False)
    except:
        dataset_val = torchvision.datasets.VOCSegmentation(root=str(dir_path), 
                                                           year='2012', 
                                                           image_set="val", 
                                                           transforms=get_transform(train=False), 
                                                           download=True)
    print("Downloaded PASCAL VOC 2012 Validation Set successfully!")
    print("------------------------------------------------------------------------------------")
    val_sampler = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, 
                                                  batch_size=BATCH_SIZE,
                                                  sampler=val_sampler, 
                                                  num_workers=WORKERS,
                                                  collate_fn=collate_fn)
    
    model.to(DEVICE)

    print("Evaluating Model on Validation Set")
    confmat = evaluate(model, data_loader_val, device=DEVICE, num_classes=20)
    confmat.compute()
    print(confmat)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Image Segmentation Benchmark')

    parser.add_argument('--model', type=str, default='fcn_resnet101', help='Select model from [fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large, lraspp_mobilenet_v3_large]')
    parser.add_argument('--pretrained', type=bool, default=True, help='Select pretraining module from [True, False]')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)