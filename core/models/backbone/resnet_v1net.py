"""ResNet models with V1Net frontend."""
import numpy as np
import torch  # pylint: disable=import-error
import torch.backends.cudnn as cudnn  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import torch.nn.functional as F  # pylint: disable=import-error
import torchvision.models as models  # pylint: disable=import-error

from .v1net import V1Net

class ResNet18_V1Net(nn.Module):
  def __init__(self, 
               timesteps=3,
               num_classes=10,
               kernel_size=5,
               kernel_size_exc=7,
               kernel_size_inh=3,
               remove_v1net=False,
               ):
    """ResNet with V1Net layer.
    params:
      timesteps: Int number of V1Net timesteps
      num_classes: Int number of output classes
      kernel_size: Int kernel size of V1Net input convolution.
      kernel_size_exc: Int kernel size for excitatory V1Net convolution.
      kernel_size_inh: Int kernel size for inhibitory V1Net convolution.
    example:
      >> x = torch.zeros(10, 3, 32, 32).cuda()
      >> resnet_v1net = ResNet18_V1Net(kernel_size=3,
                                       kernel_size_exc=7,
                                       kernel_size_inh=5).cuda()
      >> out = resnet_v1net(x)
    """
    super(ResNet18_V1Net, self).__init__()
    self.num_classes = num_classes
    self.timesteps = timesteps
    self.remove_v1net = remove_v1net
    self.rgb_mean = np.array((0.485, 0.456, 0.406)) * 255.
    self.rgb_std = np.array((0.229, 0.224, 0.225)) * 255.
    # Convert to n, c, h, w
    self.rgb_mean = self.rgb_mean.reshape((1, 3, 1, 1))
    self.rgb_mean = torch.Tensor(self.rgb_mean).float().cuda()
    self.rgb_std = self.rgb_std.reshape((1, 3, 1, 1))
    self.rgb_std = torch.Tensor(self.rgb_std).float().cuda()
    model = models.resnet18(pretrained=True).cuda()
    self.resnet_conv_1 = self.extract_layer(model, 
                                            'resnet18',
                                            'retina')
    if not self.remove_v1net:
      self.v1net_conv = V1Net(64, 64, 
                              kernel_size,
                              kernel_size_exc, 
                              kernel_size_inh)
    self.resnet_conv_2 = self.extract_layer(model,
                                            'resnet18',
                                            'cortex')
    self.fc = nn.Linear(512, num_classes)

  def standardize(self, inputs):
    """Mean normalize input images."""
    # do not normalize CIFAR-10 images, normalization added to dataloader
    return inputs
  
  def forward(self, features):
    net = self.standardize(features)
    net = self.resnet_conv_1(net)
    if not self.remove_v1net:
      n, c, h, w = net.shape
      net_tiled = net.repeat(self.timesteps, 1, 1, 1, 1).view(self.timesteps, n, c, h, w)
      net_tiled = torch.transpose(net_tiled, 1, 0)
      _, (net, _) = self.v1net_conv(net_tiled)
    net = self.resnet_conv_2(net)
    net = torch.flatten(net, 1)
    net = self.fc(net)
    return net

  def extract_layer(self, model, 
                    backbone_mode, 
                    key):
    if backbone_mode == 'resnet18':
      index_dict = {
          'retina': (0,4), 
          'cortex': (4,9),
      }
    start, end = index_dict[key]
    modified_model = nn.Sequential(*list(
      model.children()
      )[start:end])
    return modified_model





def _resnet_v1net(
    timesteps: int,
    num_classes: int,
    kernel_size: int,
    kernel_size_exc: int,
    kernel_size_inh: int,
    remove_v1net: bool,
) -> ResNet18_V1Net:
    model = ResNet18_V1Net( 
               timesteps=timesteps,
               num_classes=num_classes,
               kernel_size=kernel_size,
               kernel_size_exc=kernel_size_exc,
               kernel_size_inh=kernel_size_inh,
               remove_v1net=remove_v1net)
    return model


def resnet18_v1net(timesteps: int, num_classes: int, kernel_size: int, kernel_size_exc: int, kernel_size_inh: int, remove_v1net: bool) -> ResNet18_V1Net:
    return _resnet_v1net(timesteps=timesteps,
               num_classes=num_classes,
               kernel_size=kernel_size,
               kernel_size_exc=kernel_size_exc,
               kernel_size_inh=kernel_size_inh,
               remove_v1net=remove_v1net)
