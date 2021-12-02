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
    return presets.SegmentationPresetTrain(base_size, crop_size) if train else presets.SegmentationPresetEval(base_size)

def criterion(inputs, target):
    # classes = list(np.unique(target.detach().cpu().numpy(), return_counts=True)[0])
    # index_255 = classes.index(255)
    # classes_count = list(np.unique(target.detach().cpu().numpy(), return_counts=True)[1])
    # classes_count.pop(index_255)
    # classes.pop(index_255)

    # weights = [0.] * 21
    
    # for index in range(len(classes)):
    #     weights[classes[index]] =  max(classes_count)/classes_count[index]
    
    # weights = torch.FloatTensor(weights).cuda()

    # losses = {}

    # loss = nn.CrossEntropyLoss(weight=weights, size_average=True, ignore_index=255, reduce=True, reduction='mean')

    # for name, x in inputs.items():
    #     losses[name] = loss(input=x, target=target)
    
    # if len(losses) == 1:
    #     return losses['out']
    
    # return losses['out'] + 0.5 * losses['aux']

    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255, reduction='mean')
    if len(losses) == 1:
        return losses['out']
    return losses['out'] + 0.5 * losses['aux']

def evaluate(model, data_loader, device, num_classes, iterator):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    iou_file = str(args.output_dir) + "eval_per_image_mean_iou.csv"

    with torch.no_grad():
        start_time = time.time()
        for idx, (image, target) in enumerate(metric_logger.log_every(data_loader, 1, header)):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())
            with open(iou_file,"w") as f:
                wr = csv.writer(f,delimiter="\n")
                wr.writerow(confmat)

            if args.use_tensorboard:
                if idx != -1:
                    if ".mat" in data_loader.dataset.masks[idx]:
                        ground_truth = torch.from_numpy(scipy.io.loadmat(data_loader.dataset.masks[idx])['GTcls'][0][0][1])
                        ground_image = torch.from_numpy(mpimg.imread(data_loader.dataset.images[idx]))

                        writer.add_image('Images/val_ground_image', ground_image, iterator.eval_step, dataformats='HWC')
                        writer.add_image('Images/val_ground_truth', ground_truth, iterator.eval_step, dataformats='HW')
                        writer.add_image('Images/val_image', image[0], iterator.eval_step, dataformats='CHW')
                        writer.add_image('Images/val_target', target[0], iterator.eval_step, dataformats='HW')
                        writer.add_image('Images/val_output', get_mask(output), iterator.eval_step, dataformats='HWC')
                    else:
                        ground_truth = torch.from_numpy(mpimg.imread(data_loader.dataset.masks[idx]))
                        ground_image = torch.from_numpy(mpimg.imread(data_loader.dataset.images[idx]))

                        writer.add_image('Images/val_ground_image', ground_image, iterator.eval_step, dataformats='HWC')
                        writer.add_image('Images/val_ground_truth', ground_truth, iterator.eval_step, dataformats='HWC')
                        writer.add_image('Images/val_image', image[0], iterator.eval_step, dataformats='CHW')
                        writer.add_image('Images/val_target', target[0], iterator.eval_step, dataformats='HW')
                        writer.add_image('Images/val_output', get_mask(output), iterator.eval_step, dataformats='HWC')
                    writer.flush()

            iterator.add_eval()

        confmat.reduce_from_all_processes()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Validation time {}'.format(total_time_str))
    return confmat

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
            if args.distributed:
                div_conv_weight = model.module.backbone.div1.div.weight.data
                div_conv_weight = div_conv_weight.clamp(min=0.)
                model.module.backbone.div1.div.weight.data = div_conv_weight

                div_conv_weight = model.module.backbone.div2.div.weight.data
                div_conv_weight = div_conv_weight.clamp(min=0.)
                model.module.backbone.div2.div.weight.data = div_conv_weight

            else:
                div_conv_weight = model.module.backbone.div1.div.weight.data
                div_conv_weight = div_conv_weight.clamp(min=0.)
                model.module.backbone.div1.div.weight.data = div_conv_weight

                div_conv_weight = model.module.backbone.div2.div.weight.data
                div_conv_weight = div_conv_weight.clamp(min=0.)
                model.module.backbone.div2.div.weight.data = div_conv_weight

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

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)

    print(args)

    iterator = utils.Iterator()

    device = torch.device(args.device)

    mean_iou_list = list()
    checkpoints_list = list()

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

    model = _load_model(arch_type=args.model, 
                      backbone=args.backbone,
                      pretrained=False,
                      progress=True, 
                      num_classes=num_classes, 
                      aux_loss=args.aux_loss, 
                      divnorm_fsize=5)
    
    model.to(device)

    if args.use_load:
        checkpoint = torch.load(args.load_dir)
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

    if args.use_load:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    
    if args.test_only:
        if args.resume != '':
            checkpoint = torch.load(str(args.resume), map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes, iterator=iterator)
        print(confmat)
        return
    
    iou_file = str(args.output_dir) + ".csv"

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, args.print_freq, iterator)
                
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes, iterator=iterator)
        print(confmat)

        confmat_iu = confmat.get_IoU()
        confmat_acc_global = confmat.get_acc_global_correct()

        with open(iou_file,"w") as f:
            wr = csv.writer(f,delimiter="\n")
            wr.writerow(confmat_iu)

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
        utils.save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoint_{}.pth'.format(epoch)))
        checkpoints_list.append(checkpoint)

    max_iou_index = mean_iou_list.index(max(mean_iou_list))
    utils.save_on_master(checkpoints_list[max_iou_index], os.path.join(args.output_dir, 'best_checkpoint.pth'))
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

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
    parser.add_argument('--output-dir', default='./deeplabv3resnet50', help='path where to save')
    parser.add_argument('--use-load', dest="use_load", help="Flag to use model checkpoint", action="store_true",)
    parser.add_argument('--load-dir', default='./deeplabv3resnet50', help='path where to get model from')
    parser.add_argument('--use-tensorboard', dest="use_tensorboard", help="Flag to use tensorboard", action="store_true",)
    parser.add_argument('--tensorboard-dir', default='runs', help='path where to save tensorboard')
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
    if args.use_tensorboard:
        writer = SummaryWriter(str(args.tensorboard_dir))
    main(args)