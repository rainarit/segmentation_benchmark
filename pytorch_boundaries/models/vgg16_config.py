"""Config file for building VGG 16."""
import numpy as np

_R_MEAN = 123.68/255.
_G_MEAN = 116.78/255.
_B_MEAN = 103.94/255.
_R_STD = 0.229
_G_STD = 0.224
_B_STD = 0.225

mean_rgb_vgg = np.array([_R_MEAN, _G_MEAN, _B_MEAN])
stddev_rgb_vgg = np.array([_R_STD, _G_STD, _B_STD])


class ConfigDict(object):
  def __init__(self):
    pass


def vgg16_hed_config(model_name,
                     image_size=224,
                     num_classes=1,
                     add_v1net_early=False,
                     add_v1net=False,
                     ):
  cfg = ConfigDict()
  cfg.model_name = model_name
  cfg.image_size = image_size
  cfg.add_v1net = add_v1net
  cfg.add_v1net_early = add_v1net_early
  cfg.rgb_mean = mean_rgb_vgg
  cfg.rgb_std = stddev_rgb_vgg
  cfg.num_classes = num_classes
  return cfg