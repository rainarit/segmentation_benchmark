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
        return presets.SegmentationPresetEval(base_size=520)

def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255, reduction='mean')

    if len(losses) == 1:
        return losses["out"]

    return losses["out"] + 0.5 * losses["aux"]

def evaluate(model, data_loader, device, num_classes, root):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    image_ious = []
    with torch.no_grad():
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

def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq, scaler):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(image)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        lr_scheduler.step()

        # Clamping parameters of divnorm to non-negative values
        if "divnorm" in str(args.backbone):
            if args.distributed:
                div_conv_weight = model.module.backbone.div.div.weight.data
                div_conv_weight = div_conv_weight.clamp(min=0.)
                model.module.backbone.div.div.weight.data = div_conv_weight
            else:
                div_conv_weight = model.backbone.div.div.weight.data
                div_conv_weight = div_conv_weight.clamp(min=0.)
                model.backbone.div.div.weight.data = div_conv_weight

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

def main(args):
    if utils.is_main_process():
        results_root = Path(args.output_dir)
        output_subdir = Path("{}_{}".format(args.backbone, args.arch))
        results_root = results_root / output_subdir
        results_root.mkdir(exist_ok=True, parents=True)
        while True:
            output_root = Path("%s/%s" % (results_root, generate_rand_string(6)))
            if not os.path.exists(output_root):
                break
        output_root.mkdir(exist_ok=True, parents=True)
    
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
    print(output_root)

    device = torch.device(args.device)

    dataset, num_classes = get_dataset(args.data_path, args.dataset, "train_noval", get_transform(True, args))
    dataset_test, _ = get_dataset(args.data_path, args.dataset, "val", get_transform(False, args))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        collate_fn=utils.collate_fn,
        drop_last=True,
    )

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
        divnorm_fsize=args.divnorm_fsize,
        hidden_dim=64,
        exc_fsize=5,
        inh_fsize=3,
        timesteps=4,
        temporal_agg=False,
    )

    model.to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    params_to_optimize = [
        {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
    ]
    if args.aux_loss:
        params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})
    
    optimizer = torch.optim.SGD(
        params_to_optimize, 
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"], strict=not args.test_only)
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            if args.amp:
                scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
        print(confmat)
        return

    start_time = time.time()

    mean_ious = []
    model_checkpoints = []

    for epoch in range(args.start_epoch, args.epochs):
        epoch_dir = output_root / "epoch_{}".format(epoch)
        epoch_dir.mkdir(exist_ok=True, parents=True)

        images_dir = epoch_dir / "images"
        images_image_dir = images_dir / "image"
        images_target_dir = images_dir / "target"
        images_output_dir = images_dir / "output"
        images_dir.mkdir(exist_ok=True, parents=True)
        images_image_dir.mkdir(exist_ok=True, parents=True)
        images_target_dir.mkdir(exist_ok=True, parents=True)
        images_output_dir.mkdir(exist_ok=True, parents=True)

        checkpoints_dir = epoch_dir / "checkpoint"
        checkpoints_dir.mkdir(exist_ok=True, parents=True)

        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, args.print_freq, scaler)
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes, root=epoch_dir)
        print(confmat)
        mean_ious.append(float(confmat.get_mean_iou()))
        
        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
        }
        if args.amp:
            checkpoint["scaler"] = scaler.state_dict()
        model_checkpoints.append(checkpoint)

        utils.save_on_master(checkpoint, os.path.join(checkpoints_dir, f"model_{epoch}.pth"))

    mean_iou_file = open(output_root / 'mean_ious.csv', 'w')
    with mean_iou_file:   
        write = csv.writer(mean_iou_file)
        write.writerows(map(lambda x: [x], mean_ious))
    
    utils.save_on_master(model_checkpoints[mean_ious.index(max(mean_ious))], os.path.join(output_root, "model_best.pth"))

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
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=8, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=30, type=int, metavar="N", help="number of total epochs to run")

    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--divnorm-fsize", default=5, type=int, help="divnorm fsize"
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
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--start-epoch-checkpoint", default="", type=str)
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)