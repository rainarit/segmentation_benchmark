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
import torch.nn.functional as F
import matplotlib.image as mpimg
from coco_utils import get_coco
import presets
import utils
import os
import sys
import torch
import json
from tqdm import tqdm
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils_crf

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

class DenseCRF(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        C, H, W = probmap.shape

        U = utils_crf.unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)

        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))

        return Q

def get_dataset(dir_path, name, image_set, transform):
    def sbd(*args, **kwargs):
        return torchvision.datasets.SBDataset(*args, mode='segmentation', **kwargs)
    paths = {
        "coco": (dir_path, get_coco, 21), 
        "voc": (dir_path, torchvision.datasets.VOCSegmentation, 21),
    }
    p, ds_fn, num_classes = paths[name]
    if name == "voc":
        ds = ds_fn(p, year="2012", image_set=image_set, transforms=transform, download=False)
    else:
        ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes

def get_transform(train):
    base_size = 520
    crop_size = 480
    return presets.SegmentationPresetTrain(base_size, crop_size) if train else presets.SegmentationPresetEval(base_size)

def main(args):

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # CRF post-processor
    postprocessor = DenseCRF(
        iter_max=10,
        pos_xy_std=1,
        pos_w=3,
        bi_xy_std=67,
        bi_rgb_std=3,
        bi_w=4,
    )

    # Path to prediction images
    prediction_dir = os.path.join(
        args.output_dir,
        "features",
        "voc12",
        args.model.lower(),
        "val",
        "prediction",
    )

    # Path to logits
    logit_dir = os.path.join(
        args.output_dir,
        "features",
        "voc12",
        args.model.lower(),
        "val",
        "logit",
    )
    print("Logit src:", logit_dir)
    if not os.path.isdir(logit_dir):
        print("Logit not found, run first: python main.py test [OPTIONS]")
        quit()

    # Path to save scores
    save_dir = os.path.join(
        args.output_dir,
        "scores",
        "voc12",
        args.model.lower(),
        "val",
    )

    save_path = os.path.join(save_dir, "scores_crf.json")
    print("Score dst:", save_path)

    dataset_test, num_classes = get_dataset(args.data_path, args.dataset, "val", get_transform(train=False))

    confmat = utils.ConfusionMatrix(num_classes)

    # Process per sample
    def process(i):
        image, target = dataset_test.__getitem__(i)

        image = image.cpu().numpy()
        image = np.uint8(255 * image).transpose(1, 2, 0)

        filename = os.path.join(str(logit_dir), str(i) + ".npy")
        logit = np.load(filename)

        _, H, W = image.shape
        logit = torch.FloatTensor(logit)[None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()

        print(logit.shape)
        print(prob.shape)

        prob = postprocessor(image, logit)

        return prob, target

    for i in tqdm(range(len(dataset_test))):
        image, target = process(i)
        confmat.update(target.flatten(), image.argmax(0).flatten())
        confmat_acc_global, confmat_acc, confmat_iu = confmat.compute()
        print("Mean IoU:{}".format(confmat_iu.mean().item() * 100))
    
    confmat.reduce_from_all_processes()

    with open(save_path, "w") as f:
        print(confmat, file=f)
    

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