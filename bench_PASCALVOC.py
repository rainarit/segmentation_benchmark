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


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='fcn_resnet101', help='Select model from [fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large, lraspp_mobilenet_v3_large]')
parser.add_argument('--pretrained', type=bool, default=True, help='Select pretraining module from [True, False]')

if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    
    MODEL_NAME = args.model
    pretrained = args.pretrained
    print("Selected Model:", MODEL_NAME)
    print("Selected Pre-trained = True" if pretrained else "Selected Pre-trained = False")
    print("------------------------------------------------------------------------------------")

    model = torchvision.models.segmentation.__dict__[MODEL_NAME](num_classes=21, 
                                                                 pretrained=pretrained)

    print("Downloaded", MODEL_NAME, "successfully!")

    print("------------------------------------------------------------------------------------")

    # Inspiration from PyTorch’s GitHub repository on image segmentation transformation
    def get_transform_train(base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        transforms = []
        transforms.append(T.Resize(min_size, max_size))
        if hflip_prob > 0:
            transforms.append(T.RandomHorizontalFlip(hflip_prob))
        transforms.append(T.RandomCrop(crop_size))
        transforms.append(T.ToTensor())
        transforms.append(T.Normalize(mean=mean, std=std))
        return T.Compose(transforms)

    def get_transform_eval(base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        transforms = []
        transforms.append(T.RandomResize(base_size, base_size))
        transforms.append(T.ToTensor())
        transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))
        return T.Compose(transforms)

    def get_transform(train):
        base_size = 390
        crop_size = 360

        min_size = int((0.5 if train else 1.0) * base_size)
        max_size = int((2.0 if train else 1.0) * base_size)

        return get_transform_train(base_size, crop_size) if train else get_transform_eval(base_size)

    def cat_list(images, fill_value=0):
        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
        batch_shape = (len(images),) + max_size
        batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
        return batched_imgs


    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets
    
    def evaluate(model, data_loader, device, num_classes):
        print("---------------------Setting model to evaluation mode")
        model.eval()
        print("---------------------Generating Confusion Matrix")
        confmat = ConfusionMatrix(num_classes)
        print("---------------------Generating MetricLogger")
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Downloading PASCAL VOC 2012 Validation Set")

    dir_path = os.path.dirname(os.path.realpath(__file__))
    # Downloading PASCAL VOC 2012 Validation Set
    dataset_val = torchvision.datasets.VOCSegmentation(root=str(dir_path), 
                                                        year='2012', 
                                                        image_set="val", 
                                                        transforms=get_transform(train=False), 
                                                        download=True)

    print("Downloaded PASCAL VOC 2012 Validation Set successfully!")

    print("------------------------------------------------------------------------------------")

    val_sampler = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = torch.utils.data.DataLoader(dataset_val, 
                                                  batch_size=32,
                                                  sampler=val_sampler, 
                                                  num_workers=0,
                                                  collate_fn=collate_fn)
    
    model.to(device)

    print("Evaluating Model on Validation Set")

    confmat = evaluate(model, data_loader_val, device=device, num_classes=21)

    confmat.compute()
    print(confmat)

