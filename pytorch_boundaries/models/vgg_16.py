'''VGG11/13/16/19 in Pytorch.'''
import numpy as np
import torch  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import torchvision.models as models  # pylint: disable=import-error
import torchvision.transforms as transforms  # pylint: disable=import-error


class VGG(nn.Module):
  def __init__(self, config):
    super(VGG, self).__init__()
    self.rgb_mean = np.array((0.485, 0.456, 0.406))
    self.rgb_std = np.array((0.229, 0.224, 0.225))
    # Convert to n, c, h, w
    self.rgb_mean = self.rgb_mean.reshape((1, 3, 1, 1))
    self.rgb_mean = torch.Tensor(self.rgb_mean).float().cuda()
    self.rgb_std = self.rgb_std.reshape((1, 3, 1, 1))
    self.rgb_std = torch.Tensor(self.rgb_std).float().cuda()

    self.model = models.vgg16(pretrained=True).cuda()
  
  def standardize(self, inputs):
    """Mean normalize input images."""
    inputs = inputs - self.rgb_mean
    inputs = inputs / self.rgb_std
    return inputs

  def forward(self, inputs):
    net = self.standardize(inputs)
    net = self.model(net)
    return net
