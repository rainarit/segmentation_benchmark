
   
import datetime
import os
import time
import warnings
from pathlib import Path
import string
import numpy as np
import random

import presets
import torch
import torch.utils.data
import torchvision
import utils
from coco_utils import get_coco
from torch import nn
import transforms
from torchvision.transforms import functional as F, InterpolationMode

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

def get_transform():
    return presets.SegmentationPresetEval(base_size=520)

def evaluate(model, data_loader, device, num_classes, root):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    with torch.inference_mode():
        for index, (image, target) in enumerate(metric_logger.log_every(data_loader, 100, header)):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output["out"]
            confmat.update(target.flatten(), output.argmax(1).flatten())

            image_file = "{}/images/{}".format(root, index)
            target_file = "{}/targets/{}".format(root, index)
            output_file = "{}/outputs/{}".format(root, index)
            np.save(file=image_file, arr=image.detach().cpu().numpy())
            np.save(file=target_file, arr=target.detach().cpu().numpy())
            np.save(file=output_file, arr=output.detach().cpu().numpy())

        confmat.reduce_from_all_processes()
    return confmat

def main(args):

    results_root = Path('/home/AD/rraina/segmentation_benchmark/semseg/outputs/')
    output_subdir = Path("{}_{}_eval".format(args.backbone, args.arch))
    results_root = results_root / output_subdir
    results_root.mkdir(exist_ok=True, parents=True)
    while True:
        output_root = Path("%s/%s" % (results_root, generate_rand_string(6)))
        if not os.path.exists(output_root):
            break

    images_dir = output_root / "images"
    targets_dir = output_root / "targets"
    output_images_dir = output_root / "outputs"

    output_root.mkdir(exist_ok=True, parents=True)
    images_dir.mkdir(exist_ok=True, parents=True)
    targets_dir.mkdir(exist_ok=True, parents=True)
    output_images_dir.mkdir(exist_ok=True, parents=True)
    
    print("Output directory: {}".format(output_root))
    print("Images directory: {}".format(images_dir))
    print("Targets directory: {}".format(targets_dir))
    print("Outputs directory: {}".format(output_images_dir))

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    dataset_test, num_classes = get_dataset(args.data_path, args.dataset, "val", get_transform())

    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
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
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])

    start_time = time.time()

    confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes, root=output_root)
    print(confmat)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Evaluation time {total_time_str}")

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)
    parser.add_argument("--arch", default="deeplabv3", type=str, help="model name")
    parser.add_argument("--backbone", default="resnet50", type=str, help="backbone name")
    parser.add_argument("--aux-loss", action="store_true", help="auxiliar loss")
    parser.add_argument("--data-path", default="/home/AD/rraina/segmentation_benchmark/benchmark_RELEASE", type=str, help="dataset path")
    parser.add_argument("--dataset", default="coco", type=str, help="dataset name")
    parser.add_argument("--model", default="fcn_resnet101", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=8, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=30, type=int, metavar="N", help="number of total epochs to run")

    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)