"""Training ResNetV1Net on COCO2017."""

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

import numpy as np
import torch  # pylint: disable=import-error
import torch.backends.cudnn as cudnn  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import torch.nn.functional as F  # pylint: disable=import-error
import torch.optim as optim  # pylint: disable=import-error
import torchvision  # pylint: disable=import-error
import torchvision.models as models  # pylint: disable=import-error
import torchvision.transforms as transforms  # pylint: disable=import-error

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(os.path.split(os.path.split(cur_path)[0])[0])[0]
sys.path.append(root_path)

from segmentation_benchmark.core.models.backbone import ResNet18_V1Net
from segmentation_benchmark.core.models.backbone import PreActResNet18, PreActResNet50, PreActResNetV1Net18
from segmentation_benchmark.core.utils.coco_utils import get_coco
import segmentation_benchmark.core.utils.presets as presets
import segmentation_benchmark.core.utils.utils as utils
from segmentation_benchmark.core.models.get_segmentation_model import _segm_model
from segmentation_benchmark.core.utils.score import SegmentationMetric

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(os.path.split(os.path.split(cur_path)[0])[0])[0]
sys.path.append(root_path)

_DATASET_DIR = root_path + "/segmentation_benchmark/core/data/coco/"
_LOG_DIR = root_path + "/segmentation_benchmark/core/models/runs/logs/"

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

remove_v1net = True
test_run = False
timesteps = 0
write_results = None  # "cifar_predictions/resnet18_v1net_predictions_eval_t_%s_remove_v1net_%s" % (timesteps, remove_v1net)
checkpoint = None  # checkpoint/ckpt_4steps_reg_remove_v1net_True.pth"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.2023, 0.1994, 0.2010)),
])


trainset, num_classes = get_dataset(_DATASET_DIR, 'coco', "train", get_transform(train=True))
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=2, shuffle=True, num_workers=2)

testset, _ = get_dataset(_DATASET_DIR, 'coco', "val", get_transform(train=False))
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
# net = models.resnet18(pretrained=True)
# net = ResNet18_V1Net(kernel_size=3, kernel_size_exc=7, 
#                      kernel_size_inh=5, timesteps=timesteps,
#                      remove_v1net=remove_v1net)
net = PreActResNetV1Net18(num_classes=21, kernel_size=3, 
                          kernel_size_exc=5, kernel_size_inh=3, 
                          timesteps=3)
# net = PreActResNet18(num_classes=100)
net = net.to(device)
if device == 'cuda':
  # net = torch.nn.DataParallel(net)
  cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-2,
                       momentum=0.9, nesterov=True,
                       weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Training
def train(epoch):
  print('\nEpoch: %d' % epoch)
  net.train()
  train_loss = 0
  correct = 0
  total = 0
  for batch_idx, (inputs, targets) in enumerate(trainloader):
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    outputs = net(inputs)
    print(outputs.size())
    print(targets.size())
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()
    if not batch_idx % 50:
      print('Iter- %s, Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (batch_idx, 
                      train_loss/(batch_idx+1), 
                      100.*correct/total, 
                      correct, total))

def test(epoch):
  global best_acc
  net.eval()
  confmat = utils.ConfusionMatrix(num_classes)
  if checkpoint:
    state_dict = torch.load(checkpoint)
    print("Loading from %s with accuracy %s" % (checkpoint, 
                                                state_dict["acc"]))
    net.load_state_dict(state_dict['net'], strict=False)
  test_loss = 0
  correct = 0
  total = 0
  np_predictions = []
  np_targets = []

  with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = net(inputs)
      confmat.update(target.flatten(), output.argmax(1).flatten())
      if write_results:
        np_predictions.extend(outputs.cpu().numpy())
        np_targets.extend(targets.cpu().numpy())
      loss = criterion(outputs, targets)

      test_loss += loss.item()
      _, predicted = outputs.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()
    confmat.reduce_from_all_processes()
    print(confmat)
    print('Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = 100.*correct/total
    if write_results:
      np.array(np_predictions).dump(
            open('%s_acc_%.3f_predictions.npy' % (write_results, acc), 'wb'))
      np.array(np_targets).dump(
            open('%s_acc_%.3f_labels.npy' % (write_results, acc), 'wb'))
      return

  # Save checkpoint.
  acc = 100.*correct/total
  if acc > best_acc:
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
      os.mkdir('checkpoint')
    torch.save(state,
      './checkpoint/cifar100_ckpt_0steps_reg_remove_v1net_%s_run2.pth' % remove_v1net)
    best_acc = acc


def main():
  if test_run:
    test(-1)
  else:
    for epoch in range(start_epoch, start_epoch+200):
      train(epoch)
      test(epoch)
      if epoch > 0 and epoch % 10 == 0:
        for param_group in optimizer.param_groups:
          param_group['lr'] /= 2.

if __name__=="__main__":
  main()