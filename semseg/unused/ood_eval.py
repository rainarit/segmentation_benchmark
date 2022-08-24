import datetime
import os
import time
import csv
import numpy as np
import sys

import presets
import torch
import torch.utils.data
import torchvision
import utils
from coco_utils import get_coco
from torch import nn
import transforms

from models.segmentation.segmentation import _load_model as load_model

try:
    from torchvision import prototype
except ImportError:
    prototype = None

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

def get_transform(train, args):
    if train:
        return presets.SegmentationPresetTrain(base_size=520, crop_size=480)
    elif not args.prototype:
        return presets.SegmentationPresetEval(base_size=520)
    else:
        if args.weights:
            weights = prototype.models.get_weight(args.weights)
            return weights.transforms()
        else:
            return prototype.transforms.SemanticSegmentationEval(resize_size=520)

def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    class_iou_image = list()
    img_list = list()
    target_list = list()
    prediction_list = list()

    header = "Evaluate:"
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            
            confmat_image = utils.ConfusionMatrix(num_classes)

            output = model(image)
            output = output["out"]

            inv_normalize = transforms.Normalize(mean=(-0.485, -0.456, -0.406), std=(1/0.229, 1/0.224, 1/0.225))
            img_npy = inv_normalize(image[0], target)[0].cpu().detach().numpy()
            target_npy = target.cpu().detach().numpy()
            prediction_npy = output.cpu().detach().numpy()

            img_list.append(img_npy)
            target_list.append(target_npy)
            prediction_list.append(target_npy)


            confmat.update(target.flatten(), output.argmax(1).flatten())
            confmat_image.update(target.flatten(), output.argmax(1).flatten())
            
            class_iou_image.append(confmat_image.get_class_iou())
            confmat_image.reduce_from_all_processes()
            break

        confmat.reduce_from_all_processes()

    return confmat, class_iou_image, img_list, target_list, prediction_list

def main(args):
    if args.prototype and prototype is None:
        raise ImportError("The prototype module couldn't be found. Please install the latest torchvision nightly.")
    if not args.prototype and args.weights:
        raise ValueError("The weights parameter works only in prototype mode. Please pass the --prototype argument.")
    if args.output_dir: 
        args.output_dir = os.path.join("outputs", args.output_dir)
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    dataset_test, num_classes = get_dataset(args.data_path, args.dataset, "val", get_transform(False, args))

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    model = load_model(
        arch_type=args.arch, 
        backbone=args.backbone, 
        pretrained=args.pretrained, 
        progress=True, 
        num_classes=num_classes, 
        aux_loss=args.aux_loss, 
        divnorm_fsize=5,
    )

    model.to(device)

    model_without_ddp = model

    if args.resume:
        try:
            checkpoint = torch.load(args.resume, map_location="cpu")
            model_without_ddp.load_state_dict(checkpoint["model"])
        except:
            sys.exit('Error with checkpoint weights file.')

    confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
    print(confmat)
    return


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)

    parser.add_argument("--data-path", default="/home/AD/rraina/segmentation_benchmark/benchmark_RELEASE", type=str, help="dataset path")
    parser.add_argument("--dataset", default="voc_aug", type=str, help="dataset name")
    parser.add_argument("--arch", default="deeplabv3", type=str, help="model name")
    parser.add_argument("--backbone", default="resnet50", type=str, help="backbone name")
    parser.add_argument("--aux-loss", action="store_true", help="auxiliar loss")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=8, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=30, type=int, metavar="N", help="number of total epochs to run")

    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="/home/AD/rraina/segmentation_benchmark/semseg/outputs", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
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
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    # Prototype models only
    parser.add_argument(
        "--prototype",
        dest="prototype",
        help="Use prototype model builders instead those from main area",
        action="store_true",
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)