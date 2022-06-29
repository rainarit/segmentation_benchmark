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
from pathlib import Path
import matplotlib.image as mpimg
from coco_utils import get_coco
import presets
import utils
import os
import sys
import torch
import json
from tqdm import tqdm
import string
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

def main(args):
    utils.init_distributed_mode(args)
    args.distributed = True
    print(args)

    results_root = Path(args.output_dir)
    output_subdir = Path("{}_{}".format(args.backbone, args.arch))
    seed_subdir = Path(args.seed_dir)
    results_root = results_root / output_subdir
    results_root = results_root / seed_subdir

    # Path to images
    image_dir = results_root / "epoch_49" / "images" / "image"
    # Path to prediction images
    logit_dir = results_root / "epoch_49" / "images" / "output"
    # Path to targets
    target_dir = results_root / "epoch_49" / "images" / "target"

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

    print("Logit src:", logit_dir)

    # Path to save scores
    save_dir = results_root / "crf"
    save_dir.mkdir(exist_ok=True, parents=True)
    save_path = save_dir / "scores_crf.json"
    print("Score dst:", save_path)

    # Path to save crf images
    save_dir_images = save_dir / "images"
    save_dir_images.mkdir(exist_ok=True, parents=True)

    dataset_test, num_classes = get_dataset(args.data_path, args.dataset, "val", get_transform(False, args))

    confmat_precrf = utils.ConfusionMatrix(num_classes)
    confmat_postcrf = utils.ConfusionMatrix(num_classes)


    # Process per sample
    def process(i):
        image_filename = image_dir / str(str(i) + ".npy")
        image = torch.Tensor(np.load(image_filename))[0]

        target_filename = target_dir / str(str(i) + ".npy")
        target = torch.Tensor(np.load(target_filename))[0]

        logit_filename = logit_dir / str(str(i) + ".npy")
        logit = torch.Tensor(np.load(logit_filename))[0]

        C, H, W = image.shape

        logit = torch.FloatTensor(logit)[None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()
        image = image.numpy().astype(np.uint8).transpose(1, 2, 0)
        prob = postprocessor(image, prob)
        
        label = np.argmax(prob, axis=0)
        np.save(save_dir_images /str(str(i) + '.npy'), label)

        return image, target, logit, label

    for i in tqdm(range(len(dataset_test))):
        image, target, logit, crf = process(i)
        confmat_precrf.update(target.flatten(), logit.argmax(1).flatten())
        confmat_postcrf.update(target.flatten(), crf.flatten())
        if i % 10 == 0 and i > 0:
            print("")
            print("Pre-CRF --- ")
            print(confmat_precrf)
            print("----------------------------")
            print("Post-CRF --- ")
            print(confmat_postcrf)
    
    confmat_precrf.reduce_from_all_processes()
    confmat_postcrf.reduce_from_all_processes()

    with open(save_path, "w") as f:
        print("Pre-CRF Results: ", file=f)
        print(confmat_precrf, file=f)
        print("Post-CRF Results: ", file=f)
        print(confmat_postcrf, file=f)
    
def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Training', add_help=add_help)

    parser.add_argument("--data-path", default="/datasets01/COCO/022719/", type=str, help="dataset path")
    parser.add_argument('--seed', default=429, type=float, help='seed')
    parser.add_argument("--dataset", default="coco", type=str, help="dataset name")
    parser.add_argument("--arch", default="deeplabv3", type=str, help="model name")
    parser.add_argument("--backbone", default="resnet50", type=str, help="backbone name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--seed-dir", default=".", type=str, help="path to save seed directory")
    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)