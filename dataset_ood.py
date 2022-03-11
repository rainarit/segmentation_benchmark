import datetime
import os
import time
import csv
import shutil
import numpy as np
from tqdm import tqdm

import torch
import torch.utils.data
import torchvision
import semseg.transforms as T
import semseg.utils as utils
from torchvision.utils import save_image
from semseg.coco_utils import get_coco

class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), brightness=1, contrast=1, saturation=1, kernel_size=1, sigma=(0.1, 2.0)):
        self.transforms = T.Compose(
            [
                T.RandomResize(base_size, base_size),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
                T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation),
                #T.GaussanBlur(kernel_size, sigma),
            ]
        )

    def __call__(self, img, target):
        return self.transforms(img, target)

def get_dataset(dir_path, name, image_set, transform):
    def sbd(*args, **kwargs):
        return torchvision.datasets.SBDataset(*args, mode="segmentation", **kwargs)

    paths = {
        "voc": (dir_path, torchvision.datasets.VOCSegmentation, 21),
        "voc_aug": (dir_path, sbd, 21),
        "coco": (dir_path, get_coco, 21),
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes

def get_transform(brightness=0, contrast=0, saturation=0, kernel_size=0, sigma=(0.1, 2.0)):
    return SegmentationPresetEval(base_size=520, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), brightness=brightness, contrast=contrast, saturation=saturation, kernel_size=kernel_size, sigma=sigma)

def main(args):
    utils.init_distributed_mode(args)
    print(args)

    os.makedirs(args.root)

    if args.dataset == 'voc_aug':
        val_txt = os.path.join(args.root, 'val.txt')
        shutil.copy2('benchmark_RELEASE/dataset/val.txt', val_txt)

    print("Starting OOD dataset creation\n")
    print("="*70)

    # contrast
    print("\nOOD: Contrast\n")
    print("="*70)
    args.contrast_dir = os.path.join(args.root, "contrast")
    os.makedirs(args.contrast_dir)
    for contrast in range(1,6):
        print(f"\nContrast: {contrast}\n")
        print("="*35)
        args.contrast_severity_dir = os.path.join(args.contrast_dir, f'{contrast}')
        os.makedirs(args.contrast_severity_dir)
        dataset, _ = get_dataset(args.data_path, args.dataset, "val", get_transform(contrast=contrast))
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.workers, collate_fn=utils.collate_fn)
        for i, (image, target) in tqdm(enumerate(data_loader)):
            f = open(val_txt)
            content = f.readlines()
            save_image(image[0], str(os.path.join(args.contrast_severity_dir, f'{content[i]}.png')))
    
    # brightness
    print("\nOOD: Brightness\n")
    print("="*70)
    args.brightness_dir = os.path.join(args.root, "brightness")
    os.makedirs(args.brightness_dir)
    for brightness in range(1,6):
        print(f"\nBrightness: {brightness}\n")
        print("="*35)
        args.brightness_severity_dir = os.path.join(args.brightness_dir, f'{brightness}')
        os.makedirs(args.brightness_severity_dir)
        dataset, _ = get_dataset(args.data_path, args.dataset, "val", get_transform(brightness=brightness))
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.workers, collate_fn=utils.collate_fn)
        for i, (image, target) in enumerate(data_loader):
            f = open(val_txt)
            content = f.readlines()
            save_image(image[0], str(os.path.join(args.brightness_severity_dir, f'{content[i]}.png')))

    # gaussian blur
    # print("\nOOD: Gaussian Blur\n")
    # print("="*70)
    # args.gaussian_blur_dir = os.path.join(args.root, "gaussian_blur")
    # os.makedirs(args.gaussian_blur_dir)
    # for gaussian_blur in range(2,5):
    #     args.gaussian_blur_severity_dir = os.path.join(args.gaussian_blur_dir, f'{gaussian_blur}')
    #     os.makedirs(args.gaussian_blur_severity_dir)
    #     dataset, _ = get_dataset(args.data_path, args.dataset, "val", get_transform(gaussian_blur=gaussian_blur))
    #     data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.workers, collate_fn=utils.collate_fn)
    #     for i, (image, target) in enumerate(data_loader):
    #         f = open(val_txt)
    #         content = f.readlines()
    #         save_image(image[0], str(os.path.join(args.gaussian_blur_severity_dir, f'{content[i]}.png')))

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="OOD Dataset Creation", add_help=add_help)

    parser.add_argument("--data-path", default="/home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset", type=str, help="dataset path")
    parser.add_argument("--dataset", default="voc_aug", type=str, help="dataset name")
    parser.add_argument("--root", default="SBD_OOD", type=str, help="ood dataset name")
    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    return parser


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)