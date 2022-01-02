import transforms as T
import torch

import random
import numpy as np
from PIL import Image

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
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), contrast=1):
        self.contrast = contrast
        self.transforms = T.Compose([
            T.RandomResize(base_size, base_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            #T.ColorJitter(contrast=(contrast, contrast)),
        ])

    def oop(self, img, target, contrast=1.0):
        img_tensor, _ = T.ToTensor()(img, target)

        img_size = (img_tensor.shape[1], img_tensor.shape[2])
        portion_size = (20, 20)
        x1 = random.randint(0, img_size[0]-portion_size[0]-1)
        y1 = random.randint(0, img_size[1]-portion_size[1]-1)
        x2, y2 = x1+portion_size[0], y1+portion_size[1]
        grid = torch.clone(img_tensor[:, x1:x2, y1:y2])

        jitter = T.ColorJitter(contrast=(contrast, contrast))
        grid, _ = jitter(grid, target)

        img_tensor[:, x1:x2, y1:y2] = grid

        from torchvision import transforms
        img = transforms.ToPILImage()(img_tensor).convert("RGB")

        return img



    def __call__(self, img, target):

        img = self.oop(img, target, self.contrast)

        aug_img, target = self.transforms(img, target)

        return aug_img, target