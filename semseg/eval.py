import datetime
import os
import time
import torch
import torch.utils.data
from torch import nn
import torchvision
import numpy as np
import random
from PIL import Image
from coco_utils import get_coco
import presets
import utils
import os
import sys
import torch
import json

from tqdm import tqdm

seed=42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
g = torch.Generator()
g.manual_seed(42)

evaluate_step = 0
train_step = 0

def get_dataset(dir_path, name, image_set, transform):
    def sbd(*args, **kwargs):
        return torchvision.datasets.SBDataset(*args, mode='segmentation', **kwargs)
    paths = {
        "coco": (dir_path, get_coco, 21), 
        "voc": (dir_path, torchvision.datasets.VOCSegmentation, 21),
    }
    p, ds_fn, num_classes = paths[name]
    if name == "voc":
        ds = ds_fn(p, image_set=image_set, transforms=transform, download=True)
    else:
        ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes

def get_transform(train):
    base_size = 520
    crop_size = 480
    return presets.SegmentationPresetTrain(base_size, crop_size) if train else presets.SegmentationPresetEval(base_size)

def main(args):

    # Path to save logits
    logit_dir = os.path.join(
        args.output_dir,
        "features",
        "voc12",
        args.model.lower(),
        "val",
        "logit",
    )
    utils.mkdir(logit_dir)
    print("Logit dst:", logit_dir)

    # Path to save scores
    save_dir = os.path.join(
        args.output_dir,
        "scores",
        "voc12",
        args.model.lower(),
        "val",
    )
    utils.mkdir(save_dir)
    save_path = os.path.join(save_dir, "scores.json")
    print("Score dst:", save_path)

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)

    print(args)

    iterator = utils.Iterator()

    device = torch.device(args.device)

    dataset_test, num_classes = get_dataset(args.data_path, args.dataset, "val", get_transform(train=False))

    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)


    model = torchvision.models.segmentation.__dict__[args.model](num_classes=num_classes,
                                                                 aux_loss=args.aux_loss,
                                                                 pretrained=args.pretrained)
    model.to(torch.device('cuda'))

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
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

    checkpoint = torch.load("/home/AD/rraina/segmentation_benchmark/semseg/model_28.pth", map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    model.eval()
    
    start_time = time.time()

    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    with torch.no_grad():
        for image_id, (image, target) in enumerate(metric_logger.log_every(data_loader_test, 10, header)):
            image, target = image.to(device), target.to(device)
            logits = model(image)
            logits = logits['out']

            # Save on disk for CRF post-processing
            filename = os.path.join(str(logit_dir), str(image_id) + ".npy")
            np.save(filename, logits.cpu().numpy())

            confmat.update(target.flatten(), logits.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    with open(save_path, "w") as f:
        print(confmat, file=f)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))
    

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Training', add_help=add_help)

    parser.add_argument('--data-path', default='/home/AD/rraina/segmentation_benchmark/', help='dataset path')
    parser.add_argument('--dataset', default='coco', help='dataset name')
    parser.add_argument('--model', default='fcn_resnet101', help='model')
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
    parser.add_argument('--output-dir', default='/home/AD/rraina/segmentation_benchmark/semseg/output', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
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