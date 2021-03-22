"""Main training script for BSDS."""
import os
import re
import sys
import time

import numpy as np
import torch  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import torch.optim as optim  # pylint: disable=import-error
import torchvision  # pylint: disable=import-error
from absl import app, flags
from torch.nn import functional as F  # pylint: disable=import-error
from torch.utils.tensorboard import \
    SummaryWriter  # pylint: disable=import-error

from pytorch_boundaries.data_provider import BSDSDataProvider
from pytorch_boundaries.losses import cross_entropy_loss2d
from pytorch_boundaries.models.vgg16_config import vgg16_hed_config
from pytorch_boundaries.models.vgg_16_hed import VGG_HED
from pytorch_boundaries.models.vgg_16_hed_cam import VGG_HED_CAM

FLAGS = flags.FLAGS

flags.DEFINE_float("learning_rate", 1e-4,
                   "Learning rate")
flags.DEFINE_float("weight_decay", 1e-4,
                   "weight decay multiplier")
flags.DEFINE_float("label_gamma", 0.4,
                   "Gamma for label consensus sampling")
flags.DEFINE_float("label_lambda", 1.1,
                   "Positive weight for wce")
flags.DEFINE_boolean("use_val_for_train", False,
                     "Whether to use validation images (only after hparam search)")
flags.DEFINE_boolean("summary", True,
                     "Whether to write tensorboard summary")                     
flags.DEFINE_integer("batch_size", 1,
                   "Batch size for train/eval")
flags.DEFINE_integer("num_epochs", 15,
                     "Number of training epochs")
flags.DEFINE_integer("save_epoch", 1,
                     "Checkpoint saving frequency (in epochs)")
flags.DEFINE_integer("decay_steps", 10000,
                     "lr decay frequency (in epochs)")
flags.DEFINE_integer("v1_timesteps", 0,
                     "Number of V1Net timesteps")
flags.DEFINE_integer("update_iters", 10,
                     "Number of iterations per training loop")
flags.DEFINE_string("optimizer", "",
                    "Optimizer algorithm (Adam, SGD, etc.)")
flags.DEFINE_string("base_dir", "bsds_experiments",
                    "Base directory to store experiments")
flags.DEFINE_string("model_name", "vgg16_bn",
                    "Name of backbone network")
flags.DEFINE_string("data_dir", "",
                    "Data directory with BSDS500 images")
flags.DEFINE_string("checkpoint", "",
                    "Checkpoint file for restore and train")
flags.DEFINE_string("expt_name", "",
                    "Name of experiment w/ hyperparams")

global_step = 0

def get_params_dict(params_dict, 
                    base_lr, weight_decay):
  """Get parameter-specific training hyperparams."""
  params = []
  for k, v in params_dict.items():
    if re.match("conv_[1-4]*", k):
      if "weight" in k:
        params += [{'params': v, 
                    'lr': base_lr*1,
                    'weight_decay': weight_decay*1,
                    'name': k}]
      if "bias" in k:
        params += [{'params': v,
                    'lr': base_lr*2, 
                    'weight_decay': 0,
                    'name': k}]
    elif re.match("cam_conv[1-3]*", k):
      if "weight" in k:
        params += [{'params': v, 
                    'lr': base_lr*1,
                    'weight_decay': weight_decay*1,
                    'name': k}]
      if "bias" in k:
        params += [{'params': v,
                    'lr': base_lr*2, 
                    'weight_decay': 0,
                    'name': k}]
    elif re.match("conv_5*", k):
      if "weight" in k:
        params += [{'params': v, 
                    'lr': base_lr*100, 
                    'weight_decay': weight_decay*1,
                    'name': k}]
      if "bias" in k:
        params += [{'params': v,
                    'lr': base_lr*200, 
                    'weight_decay': 0,
                    'name': k}]
    elif re.match("dsn[1-5]_up", k):
      if "weight" in k:
        params += [{'params': v, 
                    'lr': 0, 
                    'weight_decay': 0,
                    'name': k}]
      if "bias" in k:
        params += [{'params': v,
                    'lr': 0, 
                    'weight_decay': 0,
                    'name': k}]
    elif re.match("dsn[1-5]*", k):
      if "weight" in k:
        params += [{'params': v, 
                    'lr': base_lr*0.01, 
                    'weight_decay': 1,
                    'name': k}]
      if "bias" in k:
        params += [{'params': v,
                    'lr': base_lr*0.02, 
                    'weight_decay': 0,
                    'name': k}]
    else:
      print("Learning rate for %s" % k)
      if "weight" in k:
        params += [{'params': v, 
                    'lr': base_lr*0.001, 
                    'weight_decay': 1,
                    'name': k}]
      if "bias" in k:
        params += [{'params': v,
                    'lr': base_lr*0.002, 
                    'weight_decay': 0,
                    'name': k}]
  return params

def train_epoch(model, train_dataloader,
                criterion, optimizer, writer,
                scheduler):
  """Train one epoch."""
  global global_step
  model.train()
  start_time = time.time()
  iter_loss = 0
  optimizer.zero_grad()
  loss_fun = F.binary_cross_entropy_with_logits

  for idx, data in enumerate(train_dataloader):
    # Load images and labels
    if "cam" in FLAGS.model_name:
      imgs, lbls, cam = data
      imgs, lbls, cam = imgs.float().cuda(), lbls.float().cuda(), \
                        cam.float().cuda()
      tensors = model(imgs, cam)
    else:
      imgs, lbls = data
      imgs, lbls = imgs.float().cuda(), lbls.float().cuda()
      tensors = model(imgs)

    side_output_1 = tensors["side_output_1"].float()
    side_output_3 = tensors["side_output_3"].float()
    side_output_2 = tensors["side_output_2"].float()
    side_output_4 = tensors["side_output_4"].float()
    side_output_5 = tensors["side_output_5"].float()
    fused_output = tensors["fused_output"].float()
    
    side_loss_1, _ = criterion(side_output_1, lbls, loss_fun=loss_fun)
    side_loss_2, _ = criterion(side_output_2, lbls, loss_fun=loss_fun)
    side_loss_3, _ = criterion(side_output_3, lbls, loss_fun=loss_fun)
    side_loss_4, _ = criterion(side_output_4, lbls, loss_fun=loss_fun)
    side_loss_5, _ = criterion(side_output_5, lbls, loss_fun=loss_fun)
    fused_loss, targets_confident = criterion(fused_output, lbls, loss_fun=loss_fun)
    
    total_loss = fused_loss + side_loss_1 + \
                 side_loss_2 + side_loss_3 + \
                 side_loss_4 + side_loss_5
    total_loss /= FLAGS.update_iters
    iter_loss += total_loss
    total_loss.backward()

    if not idx % FLAGS.update_iters:
      # Update parameters
      p_epoch_idx = (global_step * FLAGS.update_iters) // len(train_dataloader)
      print("Epoch(%s) - Iter (%s) - Loss: %.4f" % (p_epoch_idx,
                                                    idx/FLAGS.update_iters, 
                                                    iter_loss.item()))
      global_step += 1
      side_outputs = [side_output_1, side_output_2,
                      side_output_3, side_output_4,
                      side_output_5, fused_output
                      ]
      side_outputs = [torch.sigmoid(i) 
                      for i in side_outputs]
      if FLAGS.summary:
        if "cam" in FLAGS.model_name:
          add_summary(writer, global_step, iter_loss, 
                      imgs, lbls, side_outputs, targets_confident,
                      cam_maps=cam)
      optimizer.step()
      if not global_step % FLAGS.decay_steps:
        scheduler.step()
      optimizer.zero_grad()
      iter_loss = 0
  print("Finished training epoch in %s" % int(time.time() - start_time))


def add_summary(writer, idx, loss, 
                images, labels, outputs, labels_confident,
                cam_maps=None):
  """Write tensorboard summaries."""
  output_titles = ["side_output_%s" % i for i in range(1, 6)]
  output_titles += ["fused_prediction"]
  img_grid = torchvision.utils.make_grid(images/255)
  labels_grid = torchvision.utils.make_grid(labels)
  outputs_grid = [torchvision.utils.make_grid(output)
                  for output in outputs]
  labels_conf_grid = torchvision.utils.make_grid(labels_confident)

  writer.add_scalar("Loss/train", loss, idx)
  writer.add_scalar("Limits/Images_Max", images.max(), idx)
  writer.add_scalar("Limits/Labels_Max", labels.max(), idx)
  writer.add_image("Images/images", img_grid, idx)
  # writer.add_image("Labels/labels_unfiltered", labels_grid, idx)
  if cam_maps is not None:
    cam_grid = torchvision.utils.make_grid(cam_maps)
    writer.add_image("CAM/cam_maps", cam_grid, idx)
  writer.add_image("Labels/labels_filtered", labels_conf_grid, idx)
  for ii, (output_grid, output_title) in enumerate(zip(outputs_grid, output_titles)):
    writer.add_image("Predictions/%s" % output_title, 
                     output_grid, idx)
  return


def main(argv):
  del argv  # unused here
  device = "cuda"
  flag_args = FLAGS.flag_values_dict()

  expt_dir = os.path.join(FLAGS.base_dir, 
                          FLAGS.expt_name)
  if not os.path.exists(expt_dir):
    os.mkdir(expt_dir)
  with open(os.path.join(expt_dir, "FLAGS.txt"), "w") as f:
    f.write(str(flag_args))
    
  full_start = time.time()
  train_data = BSDSDataProvider(image_size=400,
                                is_training=True,
                                data_dir=FLAGS.data_dir,
                                cam="cam" in FLAGS.model_name)
  train_dataloader = torch.utils.data.DataLoader(train_data,
                                                 batch_size=FLAGS.batch_size, 
                                                 shuffle=True,
                                                 num_workers=FLAGS.batch_size,
                                                 )
  model_cfg = vgg16_hed_config(FLAGS.model_name, 400,
                               1, False, False)
  if "cam" in FLAGS.model_name:
    model = VGG_HED_CAM(model_cfg)
  else:
    model = VGG_HED(model_cfg)
  model.to(device)

  if FLAGS.checkpoint:
    print("Restoring from %s" % FLAGS.checkpoint)
    state_dict = torch.load(FLAGS.checkpoint)
    model.load_state_dict(state_dict)

  base_lr = FLAGS.learning_rate
  weight_decay = FLAGS.weight_decay
  params = get_params_dict(dict(model.named_parameters()),
                           base_lr, weight_decay)

  if FLAGS.optimizer.startswith("adam"):
    if weight_decay:
      optimizer = torch.optim.Adam(model.parameters(), lr=base_lr,
                                  weight_decay=weight_decay)
    else:
      print("Weight decay set to", weight_decay)
      optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
  else:
    optimizer = torch.optim.SGD(params, momentum=0.9,
                                weight_decay=weight_decay,
                                lr=base_lr)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(
              optimizer=optimizer, gamma=0.1)
  optimizer.zero_grad()
  criterion = cross_entropy_loss2d
  writer = SummaryWriter("runs/%s" % FLAGS.expt_name)
  for epoch_idx in range(FLAGS.num_epochs):
    train_epoch(model, train_dataloader,
                criterion, optimizer, writer,
                scheduler)
    if not epoch_idx % FLAGS.save_epoch:
      ckpt_dir = os.path.join(FLAGS.base_dir, 
                              FLAGS.expt_name)
      if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
      torch.save(model.state_dict(),
                 os.path.join(ckpt_dir,
                              "saved-model-epoch-%s.pth" % (epoch_idx)))
  full_duration = time.time() - full_start
  print("Training finished until" \
        "%s epochs in %s" % (FLAGS.num_epochs,
                             full_duration))
  return

if __name__=="__main__":
  app.run(main)
