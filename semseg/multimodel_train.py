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

seed=565
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

def model_obj(arch_type, backbone, dataloader, pretrained, progress, num_classes, aux_loss, divnorm_fsize, device, distributed, gpu, lr, momentum, weight_decay, resume):
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

    params_to_optimize = [
        {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
    ]

    if args.aux_loss:
        params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": lr * 10})

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=momentum, weight_decay=weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(dataloader) * args.epochs)) ** 0.9)

    if resume != '':
        checkpoint= torch.load(str(resume))
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    
    return model_without_ddp, optimizer, lr_scheduler

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

def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape

    if allkernels: 
        tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: 
        tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = torch_utils.make_grid(tensor, nrow=nrow, normalize=True, padding=2, pad_value=1.0)
    img = grid.numpy().transpose((1, 2, 0))
    return img

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
    return presets.SegmentationPresetTrain(base_size, crop_size) if train else presets.SegmentationPresetEval(base_size, contrast=args.contrast, grid_size=args.grid_size)

def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255, reduction='mean')
    if len(losses) == 1:
        return losses['out']
    return losses['out'] + 0.5 * losses['aux']

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

            break

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

def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq, iterator):

    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ") 
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)

    # Creates once at the beginning of training
    scaler = torch.cuda.amp.GradScaler()

    for idx, (image, target) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        image, target = image.to(device), target.to(device)

        # Casts operations to mixed precision
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
            for module_name, module in model.named_modules():
                    if module_name.endswith("div"):
                        curr_module_weight = module.weight.data
                        curr_module_weight = curr_module_weight.clamp(min=0.)
                        module.weight.data = curr_module_weight
            # if args.distributed:
            #     div_conv_weight = model.module.backbone.div1.div.weight.data
            #     div_conv_weight = div_conv_weight.clamp(min=0.)
            #     model.module.backbone.div1.div.weight.data = div_conv_weight

            #     div_conv_weight = model.module.backbone.div2.div.weight.data
            #     div_conv_weight = div_conv_weight.clamp(min=0.)
            #     model.module.backbone.div2.div.weight.data = div_conv_weight

            # else:
            #     div_conv_weight = model.module.backbone.div1.div.weight.data
            #     div_conv_weight = div_conv_weight.clamp(min=0.)
            #     model.module.backbone.div1.div.weight.data = div_conv_weight

            #     div_conv_weight = model.module.backbone.div2.div.weight.data
            #     div_conv_weight = div_conv_weight.clamp(min=0.)
            #     model.module.backbone.div2.div.weight.data = div_conv_weight

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        confmat_train = utils.ConfusionMatrix(21)
        confmat_train.update(target.flatten(), output['out'].argmax(1).flatten())
        confmat_train_acc_global, confmat_train_acc, confmat_train_iu = confmat_train.compute()

        if args.use_tensorboard:
            if idx % 10 == 0:
                if ".mat" in data_loader.dataset.masks[idx]:
                    ground_truth = torch.from_numpy(scipy.io.loadmat(data_loader.dataset.masks[idx])['GTcls'][0][0][1])
                    ground_image = torch.from_numpy(mpimg.imread(data_loader.dataset.images[idx]))
                    writer.add_image('Images/train_ground_image', ground_image, iterator.train_step, dataformats='HWC')
                    writer.add_image('Images/train_ground_truth', ground_truth, iterator.train_step, dataformats='HW')
                    writer.add_image('Images/train_image', image[0], iterator.train_step, dataformats='CHW')
                    writer.add_image('Images/train_target', target[0], iterator.train_step, dataformats='HW')
                    writer.add_image('Images/train_output', get_mask(output['out']), iterator.train_step, dataformats='HWC')
                else:
                    ground_truth = torch.from_numpy(mpimg.imread(data_loader.dataset.masks[idx]))
                    ground_image = torch.from_numpy(mpimg.imread(data_loader.dataset.images[idx]))
                    writer.add_image('Images/train_ground_image', ground_image, iterator.train_step, dataformats='HWC')
                    writer.add_image('Images/train_ground_truth', ground_truth, iterator.train_step, dataformats='HWC')
                    writer.add_image('Images/train_image', image[0], iterator.train_step, dataformats='CHW')
                    writer.add_image('Images/train_target', target[0], iterator.train_step, dataformats='HW')
                    writer.add_image('Images/train_output', get_mask(output['out']), iterator.train_step, dataformats='HWC')

            writer.add_scalar("Loss/train", loss.item(), iterator.train_step)
            writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], iterator.train_step)
            writer.add_scalar("Mean IoU/train", confmat_train_iu.mean().item() * 100, iterator.train_step)
            writer.add_scalar("Pixel Accuracy/train", confmat_train_acc_global.item() * 100, iterator.train_step)
            writer.flush()

        iterator.add_train()

    confmat_train.reduce_from_all_processes()

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
        num_workers=args.workers,
        collate_fn=utils.collate_fn)

    model_baseline, optimizer_baseline, lr_scheduler_baseline = model_obj(arch_type=args.model, 
                                                                        backbone=args.backbonebaseline, 
                                                                        dataloader=data_loader, 
                                                                        pretrained=False, 
                                                                        progress=True, 
                                                                        num_classes=num_classes, 
                                                                        aux_loss=args.aux_loss, 
                                                                        divnorm_fsize=5, 
                                                                        device=device, 
                                                                        distributed=args.distributed, 
                                                                        gpu=args.gpu, 
                                                                        lr=args.lr, 
                                                                        momentum=args.momentum, 
                                                                        weight_decay=args.weight_decay, 
                                                                        resume=args.resumebaseline)

    model_divnormei, optimizer_divnormei, lr_scheduler_divnormei = model_obj(arch_type=args.model, 
                                                                            backbone=args.backbonedivnormei, 
                                                                            dataloader=data_loader, 
                                                                            pretrained=False, 
                                                                            progress=True, 
                                                                            num_classes=num_classes, 
                                                                            aux_loss=args.aux_loss, 
                                                                            divnorm_fsize=5, 
                                                                            device=device, 
                                                                            distributed=args.distributed, 
                                                                            gpu=args.gpu, 
                                                                            lr=args.lr, 
                                                                            momentum=args.momentum, 
                                                                            weight_decay=args.weight_decay, 
                                                                            resume=args.resumedivnormei)
    
    if args.test_only:

        iou_file_baseline = os.path.join('/home/AD/rraina/segmentation_benchmark/semseg/csv/', str(args.outputbaseline) + "_test_test.csv")
        iou_image_file_baseline = os.path.join('/home/AD/rraina/segmentation_benchmark/semseg/csv/', str(args.outputbaseline) + "per_image_mean_test.csv")

        iou_file_divnormei = os.path.join('/home/AD/rraina/segmentation_benchmark/semseg/csv/', str(args.outputdivnormei) + "_test_test.csv")
        iou_image_file_divornmei = os.path.join('/home/AD/rraina/segmentation_benchmark/semseg/csv/', str(args.outputdivnormei) + "per_image_mean_test.csv")
        
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

        return
    
    iou_file = str(args.output) + ".csv"
    iou_image_file = str(args.output) + "per_image_mean.csv"

    start_time = time.time()

    mean_iou_list = list()

    output_dir = os.path.join('/home/AD/rraina/segmentation_benchmark/semseg/output/', args.output)
    if not(os.path.isdir(output_dir)):
        utils.mkdir(output_dir)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, args.print_freq, iterator)

        confmat, per_image_mean = evaluate(model, data_loader_test, device=device, num_classes=num_classes, iterator=iterator)
        print(confmat)

        confmat_iu = confmat.get_IoU()
        mean_iou_list.append(confmat_iu)
        confmat_acc_global = confmat.get_acc_global_correct()

        if args.use_tensorboard:
                    writer.add_scalar("Mean IoU/val", confmat_iu, epoch)
                    writer.add_scalar("Pixel Accuracy/val", confmat_acc_global, epoch)
                    writer.flush()

        checkpoint = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args
        }
        utils.save_on_master(checkpoint, os.path.join(output_dir, 'checkpoint_{}.pth'.format(epoch)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Training', add_help=add_help)

    parser.add_argument('--data-path', default='/home/AD/rraina/segmentation_benchmark/', help='dataset path')
    parser.add_argument('--seed', default=429, type=float, help='seed')
    parser.add_argument('--dataset', default='coco', help='dataset name')

    parser.add_argument('--model', default='deeplabv3', help='model')
    parser.add_argument('--backbonebaseline', default='resnet101', help='backbone')
    parser.add_argument('--backbonedivnormei', default='resnet101', help='backbone')

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

    parser.add_argument('--contrast', default=1.0, type=float)
    parser.add_argument('--grid-size', default=20, type=int)
    
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')

    parser.add_argument('--outputbaseline', default='./deeplabv3resnet50', help='path where to save')
    parser.add_argument('--outputdivnormei', default='./deeplabv3resnet50', help='path where to save')

    parser.add_argument('--use-tensorboard', dest="use_tensorboard", help="Flag to use tensorboard", action="store_true",)
    parser.add_argument('--tensorboard-dir', default='runs', help='path where to save tensorboard')

    parser.add_argument('--resumebaseline', default='', help='resume from checkpoint baseline')
    parser.add_argument('--resumedivnormei', default='', help='resume from checkpoint divnormei')

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
    if args.use_tensorboard:
        writer = SummaryWriter(str(args.tensorboard_dir))
    main(args)
