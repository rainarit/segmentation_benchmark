import transforms as T
import torch

import random
import numpy as np
from PIL import Image
import os

from torchvision.utils import save_image

class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        aug_img, target = self.transforms(img, target)

        return aug_img, target

class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), contrast=1, brightness=1, hue=1, sigma=1, kernel_size=1):
        self.contrast_initial = contrast
        self.contrast = contrast
        self.brightness = brightness
        self.hue = hue
        self.sigma = sigma
        self.kernel_size = kernel_size

        if self.contrast != 1.0:
            self.contrast_initial=self.contrast-1.0
        else:
            self.contrast_initial=1.0
        
        if self.sigma != 1.0:
            self.sigma_initial=self.sigma-1.0
        else:
            self.sigma_initial=1.0

        if self.brightness != 1.0:
            self.brightness_initial=self.brightness-1.0
        else:
            self.brightness_initial=1.0

        if self.hue != -0.5:
            self.hue_initial=self.hue-0.1
        else:
            self.hue_initial=-0.5

        print("Contrast: ({}, {})".format(str(self.contrast_initial), str(self.contrast)))
        print("Brightness: ({}, {})".format(str(self.brightness_initial), str(self.brightness)))
        print("Kernel Size: {}".format(self.kernel_size))
        print("Sigma: ({}, {})".format(str(self.sigma_initial), str(self.sigma)))

        self.transforms = T.Compose([
            #T.RandomResize(base_size, base_size),
            T.RandomResize(base_size, base_size),
            #T.ColorJitter(brightness=(self.brightness_initial, self.brightness)),
            #T.GaussianBlur(kernel_size=self.kernel_size, sigma=self.sigma),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])



    def __call__(self, img, target):

        aug_img, target = self.transforms(img, target)

        return aug_img, target