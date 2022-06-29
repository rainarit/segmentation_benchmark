import datetime
import os
import time
import warnings
import numpy as np
from pathlib import Path
import string
import random
import glob 
import csv 
from os.path import dirname

import presets
import torch
import torch.utils.data
import torchvision
import utils
from coco_utils import get_coco
from torch import nn

from models.segmentation.segmentation import _load_model as load_model

def generate_rand_string(n):
  letters = string.ascii_lowercase
  str_rand = ''.join(random.choice(letters) for i in range(n))
  return str_rand

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

    else:
        return presets.SegmentationPresetEval(base_size=520,
                                                jitter=args.jitter, 
                                                contrast_min=args.contrast_min,
                                                contrast_max=args.contrast_max,
                                                brightness_min=args.brightness_min,
                                                brightness_max=args.brightness_max,
                                                occlude_min=args.occlude_min, 
                                                occlude_max=args.occlude_max,
                                              )

def evaluate(model, data_loader, device, num_classes, root):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    image_ious = []
    with torch.inference_mode():
        for index, (image, target) in enumerate(metric_logger.log_every(data_loader, 100, header)):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output["out"]

            confmat.update(target.flatten(), output.argmax(1).flatten())

            np.save(file=root / "images" / "image" / str(index), arr=image.detach().cpu().numpy())
            np.save(file=root / "images" / "target" / str(index), arr=target.detach().cpu().numpy())
            np.save(file=root / "images" / "output" / str(index), arr=output.detach().cpu().numpy())

            image_ious.append(confmat.get_class_iou())

    image_iou_file = open(root / 'image_mean_ious.csv', 'w')
    with image_iou_file:   
        write = csv.writer(image_iou_file)
        write.writerows(image_ious)

        confmat.reduce_from_all_processes()

    return confmat

def main(args):
    results_root = Path(dirname(args.resume))
    if args.contrast_max > 0:
        args.pertub = 'contrast'
        results_root = results_root / "jitter" / str(args.pertub) /"({},{})".format(args.contrast_min, args.contrast_max)
    if args.brightness_max > 0:
        args.pertub = 'brightness'
        results_root = results_root / "jitter" / str(args.pertub) /"({},{})".format(args.brightness_min, args.brightness_max)
    if args.occlude_max > 1.0:
        args.pertub = 'occlude'
        results_root = results_root / "jitter" / str(args.pertub) /"({},{})".format(args.occlude_min, args.occlude_max)
    results_root.mkdir(exist_ok=True, parents=True)

    print("Result stored dir: ", results_root)
    
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

    dataset_test, num_classes = get_dataset(args.data_path, args.dataset, "val", get_transform(False, args))

    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=1, 
        sampler=test_sampler, 
        num_workers=args.workers, 
        collate_fn=utils.collate_fn
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
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])

    confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes, root=results_root)
    start_time = time.time()

    print(confmat)

    mean_iou_file = open(results_root / 'mean_ious.csv', 'w')
    with mean_iou_file:   
        write = csv.writer(mean_iou_file)
        write.writerows([float(confmat.get_mean_iou())])
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)

    parser.add_argument("--data-path", default="/datasets01/COCO/022719/", type=str, help="dataset path")
    parser.add_argument('--seed', default=429, type=float, help='seed')
    parser.add_argument("--dataset", default="coco", type=str, help="dataset name")
    parser.add_argument("--arch", default="deeplabv3", type=str, help="model name")
    parser.add_argument("--backbone", default="resnet50", type=str, help="backbone name")
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    parser.add_argument("--aux-loss", action="store_true", help="auxiliar loss")
    parser.add_argument("--jitter", default="contrast", type=str, help="jitter name")
    parser.add_argument("--contrast-min", default=1.0, type=float, help="contrast_min")
    parser.add_argument("--contrast-max", default=1.0, type=float, help="contrast_max")
    parser.add_argument("--brightness-min", default=1.0, type=float, help="brightness_min")
    parser.add_argument("--brightness-min", default=1.0, type=float, help="brightness_max")
    parser.add_argument("--occlude-min", default=0.0, type=float, help="occlude_min")
    parser.add_argument("--occlude-max", default=0.0, type=float, help="occlude_max")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")

    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)