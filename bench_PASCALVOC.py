from __future__ import print_function
from collections import defaultdict, deque
import datetime
import math
import time
import torch
import torch.distributed as dist
from torchvision import transforms as T
from torchvision.transforms import functional as F
import torch.utils.data
from torch import nn
import torchvision
import torchbench
from torchbench.semantic_segmentation.transforms import (
    Normalize,
    Resize,
    ToTensor,
    Compose,
)
import numpy as np
from PIL import Image
import random
import errno
import os
import pathlib
import argparse

from MetricLogger import MetricLogger
from SmoothedValue import SmoothedValue
from ConfusionMatrix import ConfusionMatrix

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='fcn_resnet101', help='Choose from following models : [fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large, lraspp_mobilenet_v3_large]')
parser.add_argument('--pretrained', type=bool, default=true, help='Choose from following : [true, false]')
args = parser.parse_args()

def get_transform(train):
    base_size = 520
    crop_size = 480

    min_size = int((0.5 if train else 1.0) * base_size)
    max_size = int((2.0 if train else 1.0) * base_size)
    transforms = []
    transforms.append(Resize((520, 480)))
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))
    return Compose(transforms)
    


def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = ConfusionMatrix(num_classes)
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

MODEL_NAME = args.model
print('Downloading ', MODEL_NAME)
model = torchvision.models.segmentation.__dict__[MODEL_NAME](num_classes=21, pretrained=True)
print('Downloaded ', MODEL_NAME)

device = torch.device('cpu')
dataset_test = torchvision.datasets.VOCSegmentation(root=str(pathlib.Path().absolute()), year='2012', image_set="val", transforms=get_transform(train=False), download=True)
test_sampler = torch.utils.data.SequentialSampler(dataset_test)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=32,
    sampler=test_sampler, num_workers=4,
    collate_fn=collate_fn)
model.to(device)
confmat = evaluate(model, data_loader_test, device=device, num_classes=21)

confmat.compute()
print(confmat)