import datetime
import os
import time
import torch
import torch.utils.data
import transforms as T
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

from torchvision.utils import save_image
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from models.segmentation.segmentation import _load_model as load_model
import  csv
from torchvision.utils import save_image


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
    return presets.SegmentationPresetTrain(base_size, crop_size) if train else presets.SegmentationPresetEval(base_size, contrast=args.contrast, 
                                                                                                              brightness=args.brightness, sigma=args.sigma, 
                                                                                                              occlusion=args.occlude, jitter=False, blur=False, 
                                                                                                              occlude=True)

def evaluate(model, data_loader, device, num_classes, output_dir, save=False):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    confmat = utils.ConfusionMatrix(num_classes)
    per_mean_iou = list()

    image_epoch_dir = os.path.join(output_dir, "images/")
    utils.mkdir(image_epoch_dir)
    image_dir = os.path.join(output_dir, "images/image/")
    utils.mkdir(image_dir)
    target_dir = os.path.join(output_dir, "images/target/")
    utils.mkdir(target_dir)
    prediction_dir = os.path.join(output_dir, "images/prediction/")
    utils.mkdir(prediction_dir)
    mean_iou_file = os.path.join(output_dir, "mean_iou.csv")
    per_mean_iou_file = os.path.join(output_dir, "per_image_mean_iou.csv")

    with torch.no_grad():
        start_time = time.time()
        for idx, (image, target) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            image, target = image.to(device), target.to(device)
            
            confmat_image = utils.ConfusionMatrix(num_classes)

            output = model(image)
            output = output['out']

            if save:
                image_path =  os.path.join(image_dir, '{}.npy'.format(idx))
                target_path = os.path.join(target_dir, '{}.npy'.format(idx))
                prediction_path = os.path.join(prediction_dir, '{}.npy'.format(idx))

                inv_normalize = T.Normalize(mean=(-0.485, -0.456, -0.406), std=(1/0.229, 1/0.224, 1/0.225))
                img_npy = inv_normalize(image[0], target)[0].cpu().detach().numpy()
                target_npy = target.cpu().detach().numpy()
                prediction_npy = output.cpu().detach().numpy()

                utils.save_np_image(image_path, img_npy)
                utils.save_np_image(target_path, target_npy)
                utils.save_np_image(prediction_path, prediction_npy)

            confmat.update(target.flatten(), output.argmax(1).flatten())

            confmat_image.update(target.flatten(), output.argmax(1).flatten())
            acc_global, acc, iu = confmat_image.compute()
            confmat_image.reduce_from_all_processes()

            image_mean_iou = list((iu * 100).tolist())
            per_mean_iou.append(image_mean_iou)

        confmat.reduce_from_all_processes()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Validation time {}'.format(total_time_str))
    
    with open(per_mean_iou_file, 'w', newline='') as f:
            wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            wr.writerow(per_mean_iou)
    with open(mean_iou_file, 'w', newline='') as f:
            wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            wr.writerow([confmat.get_mean_iou()])
    return confmat

def Model(arch_type, backbone, num_classes, divnorm_fsize, checkpoint, distributed, gpu, device, pretrained=False, progress=True, aux_loss=True):
    model = load_model(arch_type=arch_type, 
                       backbone=backbone,
                       pretrained=pretrained,
                       progress=progress, 
                       num_classes=num_classes, 
                       aux_loss=aux_loss, 
                       divnorm_fsize=divnorm_fsize)

    model.to(device)

    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    params_to_optimize = [
        {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
    ]

    if aux_loss:
        params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})
    
    checkpoint = torch.load(str(checkpoint), map_location='cuda:0')
    model_without_ddp.load_state_dict(checkpoint['model'])

    #try:
    #    checkpoint = torch.load(str(checkpoint))
    #    model_without_ddp.load_state_dict(checkpoint['model'])
    #except:
    #    sys.exit('Error with checkpoint weights file.')

    print("=> Created model")
    print("==> Arch Type: " + arch_type)
    print("==> Backbone Type: " + backbone)

    return model_without_ddp

def main(args):

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
    args.distributed=False

    print(args)

    device = torch.device(args.device)

    dataset_test, num_classes = get_dataset(args.data_path, args.dataset, "val", get_transform(train=False))

    data_loader_test = torch.utils.data.DataLoader(dataset_test, 
        batch_size=1, 
        num_workers=args.workers, 
        collate_fn=utils.collate_fn)
    
    for idx, backbone in enumerate(args.backbone):
        model = Model(arch_type=args.model,
                      backbone=backbone, 
                      num_classes=num_classes, 
                      divnorm_fsize=5, 
                      checkpoint=args.checkpoint[idx], 
                      distributed=args.distributed, 
                      gpu=args.gpu, 
                      device=device, 
                      pretrained=False, 
                      progress=True, 
                      aux_loss=True)
        output_dir = os.path.join('/home/AD/rraina/segmentation_benchmark/semseg/outputs/', args.output[idx])
        print(output_dir)
        output_val_test_dir = os.path.join(output_dir, "val_test_only/")
        if not(os.path.isdir(output_dir)):
            utils.mkdir(output_dir)
        if not(os.path.isdir(output_val_test_dir)):
            utils.mkdir(output_val_test_dir)
        mean_iou_file = os.path.join(output_val_test_dir, "mean_iou.csv")  

        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes, output_dir=output_val_test_dir, save=True) 

        print(confmat)

    return

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Training', add_help=add_help)

    parser.add_argument('--data-path', default='/home/AD/rraina/segmentation_benchmark/', help='dataset path')
    parser.add_argument('--seed', default=429, type=float, help='seed')
    parser.add_argument('--dataset', default='coco', help='dataset name')
    parser.add_argument('--model', default='deeplabv3', help='model')
    parser.add_argument('--backbone', default=['resnet101'], help='backbone', nargs='+')
    parser.add_argument('--aux-loss', action='store_true', help='auxiliar loss')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--contrast', default=1.0, type=float)
    parser.add_argument('--brightness', default=1.0, type=float)
    parser.add_argument('--hue', default=1.0, type=float)
    parser.add_argument('--kernel_size', default=1, type=int)
    parser.add_argument('--sigma', default=1.0, type=float)
    parser.add_argument('-ol', '--occlude', default=0., type=float, help='range for occlusion perturbation')
    parser.add_argument('--print-freq', default=800, type=int, help='print frequency')
    parser.add_argument('--output', default=['deeplabv3resnet50'], help='path where to save', nargs='+')
    parser.add_argument('--checkpoint', default=['deeplabv3resnet50'], help='resume from checkpoint', nargs='+')
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
