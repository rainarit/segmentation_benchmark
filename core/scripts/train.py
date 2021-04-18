import datetime
import os
import time
import shutil

import torch
import torch.utils.data
from torch import nn
import torchvision
import os
import sys


cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(os.path.split(os.path.split(cur_path)[0])[0])[0]
sys.path.append(root_path)

from segmentation_benchmark.core.utils.coco_utils import get_coco
import segmentation_benchmark.core.utils.presets as presets
import segmentation_benchmark.core.utils.utils as utils
from segmentation_benchmark.core.models.get_segmentation_model import _segm_model
from segmentation_benchmark.core.utils.score import SegmentationMetric

def get_dataset(dir_path, name, image_set, transform):
    def sbd(*args, **kwargs):
        return torchvision.datasets.SBDataset(*args, mode='segmentation', **kwargs)
    paths = {
        "coco": (dir_path, get_coco, 21)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train):
    base_size = 520
    crop_size = 480

    return presets.SegmentationPresetTrain(base_size, crop_size) if train else presets.SegmentationPresetEval(base_size)


def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}_{}.pth'.format(args.model, args.backbone, args.dataset)
    filename = os.path.join(directory, filename)

    if args.distributed:
        model = model.module
    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = '{}_{}_{}_best_model.pth'.format(args.model, args.backbone, args.dataset)
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
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


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    # setup logger-2
    logger = utils.setup_logger("semantic_segmentation", args.log_dir, utils.get_rank(), filename='{}_{}_{}_log.txt'.format(
        args.model, args.backbone, args.dataset))
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)


    # dataset and dataloader
    dataset, num_classes = get_dataset(args.data_path, args.dataset, "train", get_transform(train=True))
    dataset_test, _ = get_dataset(args.data_path, args.dataset, "val", get_transform(train=False))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)


    # create network
    model = _segm_model(name=args.model, 
                        backbone_name=args.backbone, 
                        num_classes=num_classes, 
                        aux=args.aux_loss, 
                        pretrained_backbone=True)
    model.to(device)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module


    # optimizer
    params_to_optimize = [
        {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
    ]

    if args.aux_loss:
        params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})

    optimizer = torch.optim.SGD(params_to_optimize,
                                lr=args.lr, 
                                momentum=args.momentum, 
                                weight_decay=args.weight_decay)


    # lr scheduling
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    # evaluation metrics
    metric = SegmentationMetric(num_classes)

    best_pred = 0.0


    # resume checkpoint if needed
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=not args.test_only)
        if not args.test_only:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
        print(confmat)
        return


    # training
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        print_freq = args.print_freq

        model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
        header = 'Epoch: [{}]'.format(epoch)
        for iteration, (image, target) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            image, target = image.to(device), target.to(device)
            output = model(image)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        # evaluation   
        is_best = False
        metric.reset() 
        model.eval()
        confmat = utils.ConfusionMatrix(num_classes)
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test:'
        with torch.no_grad():
            for i, (image, target) in enumerate(metric_logger.log_every(data_loader_test, 100, header)):
                image, target = image.to(device), target.to(device)
                output = model(image)
                output = output['out']

                metric.update(output, target)
                confmat.update(target.flatten(), output.argmax(1).flatten())

                pixAcc, mIoU = metric.get()
                logger.info("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc, mIoU))
            
            new_pred = (pixAcc + mIoU) / 2
            if new_pred > best_pred:
                is_best = True
                best_pred = new_pred
            save_checkpoint(model, args, is_best)

            confmat.reduce_from_all_processes()
        print(confmat)

        utils.save_on_master(
            {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args
            },
            os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Training')

    parser.add_argument('--data-path', default='segmentation_benchmark/core/data/coco/', help='dataset path')
    parser.add_argument('--dataset', default='coco', help='dataset name')
    parser.add_argument('--model', default='fcn', help='model')
    parser.add_argument('--backbone', default='resnet50', help='backbone')
    parser.add_argument('--aux-loss', action='store_true', help='auxiliar loss')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--log-dir', default='segmentation_benchmark/core/models/runs/logs/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--log-iter', type=int, default=10,
                        help='print log every log-iter')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-dir', default='segmentation_benchmark/core/models',
                        help='Directory for saving checkpoint models')
    # evaluation only
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
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
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)