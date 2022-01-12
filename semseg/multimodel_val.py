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
from models.segmentation.segmentation import _load_model
import  csv
from torchvision.utils import save_image

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

evaluate_step = 0

def model(arch_type, backbone, pretrained, progress, num_classes, aux_loss, divnorm_fsize, device, distributed, gpu, resume):
    model = _load_model(arch_type=arch_type, 
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

    checkpoint= torch.load(str(resume))
    model_without_ddp.load_state_dict(checkpoint['model'])
    
    return model_without_ddp

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

def get_mask(output):
    output_predictions = output[0].argmax(0)
    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy())
    r.putpalette(colors)
    return np.array(r.convert('RGB'))

def get_transform(train):
    base_size = 520
    crop_size = 480
    return presets.SegmentationPresetTrain(base_size, crop_size) if train else presets.SegmentationPresetEval(base_size, contrast=args.contrast)

def evaluate(model_baseline, model_compare, data_loader, device, num_classes, iterator, output_folder, output_folder_compare, save=True):
    model_baseline.eval()
    model_compare.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    confmat_baseline = utils.ConfusionMatrix(num_classes)
    per_mean_iou_baseline = list()

    confmat_compare = utils.ConfusionMatrix(num_classes)
    per_mean_iou_compare = list()

    with torch.no_grad():
        start_time = time.time()
        for idx, (image, target) in enumerate(metric_logger.log_every(data_loader, 1, header)):
            image, target = image.to(device), target.to(device)

            confmat_image_baseline = utils.ConfusionMatrix(num_classes)
            confmat_image_compare = utils.ConfusionMatrix(num_classes)

            output_baseline = model_baseline(image)
            output_baseline = output_baseline['out']

            output_compare = model_compare(image)
            output_compare = output_compare['out']

            inv_normalize = T.Normalize(mean=(-0.485, -0.456, -0.406), std=(1/0.229, 1/0.224, 1/0.225))

            print(inv_normalize(image[0], target).shape)

            if save==True:
                image_path = '/home/AD/rraina/segmentation_benchmark/semseg/images/' + str(output_folder) + "/val/image/"
                target_path = '/home/AD/rraina/segmentation_benchmark/semseg/images/' + str(output_folder) + "/val/target/"
                output_path = '/home/AD/rraina/segmentation_benchmark/semseg/images/' + str(output_folder) + "/val/output/"
                utils.mkdir(image_path)
                utils.mkdir(target_path)
                utils.mkdir(output_path)

                inv_normalize = T.Normalize(mean=(-0.485, -0.456, -0.406), std=(1/0.229, 1/0.224, 1/0.225))

                save_image(inv_normalize(image[0], target)[0], image_path + str(idx) + ".png")
                save_image(target[0].float(), target_path+ str(idx) + ".png")
                save_image(torch.from_numpy(get_mask(output_baseline)).float(), output_path+ str(idx) + ".png")

                image_path = '/home/AD/rraina/segmentation_benchmark/semseg/images/' + str(output_folder_compare) + "/val/image/"
                target_path = '/home/AD/rraina/segmentation_benchmark/semseg/images/' + str(output_folder_compare) + "/val/target/"
                output_path = '/home/AD/rraina/segmentation_benchmark/semseg/images/' + str(output_folder_compare) + "/val/output/"
                utils.mkdir(image_path)
                utils.mkdir(target_path)
                utils.mkdir(output_path)

                save_image(inv_normalize(image[0], target)[0], image_path + str(idx) + ".png")
                save_image(target[0].float(), target_path+ str(idx) + ".png")
                save_image(torch.from_numpy(get_mask(output_compare)).float(), output_path+ str(idx) + ".png")

            confmat_baseline.update(target.flatten(), output_baseline.argmax(1).flatten())
            confmat_compare.update(target.flatten(), output_compare.argmax(1).flatten())

            confmat_image_baseline.update(target.flatten(), output_baseline.argmax(1).flatten())
            acc_global_baseline, acc_baseline, iu_baseline = confmat_image_baseline.compute()
            confmat_image_baseline.reduce_from_all_processes()

            image_mean_iou = list((iu_baseline * 100).tolist())
            per_mean_iou_baseline.append(image_mean_iou)

            confmat_image_compare.update(target.flatten(), output_compare.argmax(1).flatten())
            acc_global_compare, acc_compare, iu_compare = confmat_image_compare.compute()
            confmat_image_compare.reduce_from_all_processes()

            image_mean_iou = list((iu_compare * 100).tolist())
            per_mean_iou_compare.append(image_mean_iou)

            iterator.add_eval()

        confmat_baseline.reduce_from_all_processes()
        confmat_compare.reduce_from_all_processes()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Validation time {}'.format(total_time_str))
    return confmat_baseline, per_mean_iou_baseline, confmat_compare, per_mean_iou_compare

def main(args):

    utils.init_distributed_mode(args)

    print(args)

    iterator = utils.Iterator()

    device = torch.device(args.device)

    dataset_test, num_classes = get_dataset(args.data_path, args.dataset, "val", get_transform(train=False))

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        num_workers=args.workers,
        collate_fn=utils.collate_fn)

    model_baseline = model(arch_type=args.model, 
                            backbone=args.backbonebaseline, 
                            pretrained=False, 
                            progress=True, 
                            num_classes=num_classes, 
                            aux_loss=args.aux_loss, 
                            divnorm_fsize=5, 
                            device=device, 
                            distributed=args.distributed, 
                            gpu=args.gpu,
                            resume=args.resumebaseline)

    model_divnormei = model(arch_type=args.model, 
                            backbone=args.backbonedivnormei, 
                            pretrained=False, 
                            progress=True, 
                            num_classes=num_classes, 
                            aux_loss=args.aux_loss, 
                            divnorm_fsize=5, 
                            device=device, 
                            distributed=args.distributed, 
                            gpu=args.gpu, 
                            resume=args.resumedivnormei)

    val_path = '/home/AD/rraina/segmentation_benchmark/semseg/output/' + str(args.outputcompare) + "/val/"
    utils.mkdir(val_path)   


    
     

    iou_file_baseline = os.path.join('/home/AD/rraina/segmentation_benchmark/semseg/csv/', str(args.outputbaseline) + "_test_test.csv")
    iou_image_file_baseline = os.path.join('/home/AD/rraina/segmentation_benchmark/semseg/csv/', str(args.outputbaseline) + "per_image_mean.csv")

    iou_file_divnormei = os.path.join('/home/AD/rraina/segmentation_benchmark/semseg/csv/', str(args.outputcompare) + "_test_test.csv")
    iou_image_file_divornmei = os.path.join('/home/AD/rraina/segmentation_benchmark/semseg/csv/', str(args.outputcompare) + "per_image_mean.csv")
    
    confmat_baseline, per_image_mean_baseline, confmat_divnormei, per_image_mean_divnormei = evaluate(model_baseline, model_divnormei, data_loader_test, device=device, num_classes=num_classes, iterator=iterator, output_folder=args.outputbaseline, output_folder_compare=args.outputdivnormei)
    confmat_iu_baseline = confmat_baseline.get_IoU()
    confmat_iu_divnormei = confmat_divnormei.get_IoU()

    writer_baseline=csv.writer(open(iou_file_baseline,'w'))
    print("Baseline Model Mean IoU: {}%".format(confmat_iu_baseline))
    print(confmat_baseline)
    writer_baseline.writerow([confmat_baseline])

    writer_per_image_mean_baseline=csv.writer(open(iou_image_file_baseline,'w'))
    for image_mean_iou in per_image_mean_baseline:
        writer_per_image_mean_baseline.writerow([image_mean_iou])

    writer_divnormei=csv.writer(open(iou_file_divnormei,'w'))
    print("DivNormEI Model Mean IoU: {}%".format(confmat_iu_divnormei))
    print(confmat_divnormei)
    writer_divnormei.writerow([confmat_divnormei])
    
    writer_per_image_mean_divnormei=csv.writer(open(iou_image_file_divornmei,'w'))
    for image_mean_iou in per_image_mean_divnormei:
        writer_per_image_mean_divnormei.writerow([image_mean_iou])


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Training', add_help=add_help)

    parser.add_argument('--backbonebaseline', default='resnet101', help='backbone')
    parser.add_argument('--backbonecompare', default='resnet101', help='backbone')
    parser.add_argument('--outputbaseline', default='./deeplabv3resnet50', help='path where to save')
    parser.add_argument('--outputcompare', default='./deeplabv3resnet50', help='path where to save')
    parser.add_argument('--resumebaseline', default='', help='resume from checkpoint baseline model')
    parser.add_argument('--resumecompare', default='', help='resume from checkpoint compare model')
    
    parser.add_argument('--data-path', default='/home/AD/rraina/segmentation_benchmark/', help='dataset path')
    parser.add_argument('--seed', default=429, type=float, help='seed')
    parser.add_argument('--dataset', default='coco', help='dataset name')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--contrast', default=1.0, type=float)

    parser.add_argument('--model', default='deeplabv3', help='model')
    parser.add_argument('--aux-loss', action='store_true', help='auxiliar loss')
    parser.add_argument('--device', default='cuda', help='device')

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
