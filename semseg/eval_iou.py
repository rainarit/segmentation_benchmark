import datetime
import os
import time
import torch
import torch.utils.data
from torch import nn
import torchvision
import numpy as np
import scipy.io
import random
from PIL import Image
import matplotlib.image as mpimg
from tqdm import tqdm
from coco_utils import get_coco
import presets
import utils
from torchvision import utils as torch_utils
import os
import sys
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import math
from torch.utils.tensorboard import SummaryWriter
from models.segmentation.segmentation import _load_model
import ipdb
import  csv

seed=429
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

evaluate_step = 0
train_step = 0

def get_dataset(dir_path, name, image_set, transform):
    def sbd(*args, **kwargs):
        return torchvision.datasets.SBDataset(*args, mode='segmentation', **kwargs)
    paths = {
        "coco": (dir_path, get_coco, 21), 
        "voc": (dir_path, torchvision.datasets.VOCSegmentation, 21),
        "voc_aug": (dir_path, sbd, 21)
    }
    p, ds_fn, num_classes = paths[name]
    if name == "voc":
        ds = ds_fn(p, year="2012", image_set=image_set, transforms=transform, download=False)
    elif name == "voc_aug":
        ds = ds_fn(p, image_set=image_set, transforms=transform, download=False)
    else:
        ds = ds_fn(p, image_set=image_set, transforms=transform, download=False)
    return ds, num_classes

def get_transform(train):
    base_size = 520
    crop_size = 480
    return presets.SegmentationPresetTrain(base_size, crop_size) if train else presets.SegmentationPresetEval(base_size)

def evaluate(model, csvwriter, data_loader, device, num_classes, iterator):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    with torch.no_grad():
        start_time = time.time()
        for idx, (image, target) in enumerate(metric_logger.log_every(data_loader, 1, header)):

            confmat = utils.ConfusionMatrix(num_classes)
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())

            acc_global, acc, iu = confmat.compute()

            csvwriter.writerow(list((iu * 100).tolist()))

            iterator.add_eval()

            confmat.reduce_from_all_processes()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Validation time {}'.format(total_time_str))

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main(args):

    utils.init_distributed_mode(args)

    print(args)

    iterator = utils.Iterator()

    device = torch.device(args.device)

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

    iou_file = str(args.output_dir) + "eval_per_image_mean_iou.csv"

    with open(iou_file, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile)

        for i in range(0, 50):
            model = _load_model(arch_type=args.model, 
                        backbone=args.backbone,
                        pretrained=False,
                        progress=True, 
                        num_classes=num_classes, 
                        aux_loss=args.aux_loss, 
                        divnorm_fsize=5)
        
            model.to(device)

            checkpoint_path = args.output_dir + "/checkpoint_" + str(i) + ".pth"
            print(checkpoint_path)
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model'])


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
                lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            evaluate(model, csvwriter, data_loader_test, device=device, num_classes=num_classes, iterator=iterator)

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Training', add_help=add_help)

    parser.add_argument('--data-path', default='/home/AD/rraina/segmentation_benchmark/', help='dataset path')
    parser.add_argument('--dataset', default='coco', help='dataset name')
    parser.add_argument('--model', default='deeplabv3', help='model')
    parser.add_argument('--backbone', default='resnet101', help='backbone')
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
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--load-dir', default='./deeplabv3resnet50', help='path where to get model from')
    parser.add_argument('--output-dir', default='./deeplabv3resnet50', help='path where to save')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
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

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)