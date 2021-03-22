"""VGG16-HED implementation."""
import numpy as np
import torch  # pylint: disable=import-error
import torch.nn as nn # pylint: disable=import-error
import torchvision.transforms as transforms  # pylint: disable=import-error
import torchvision.models as models  # pylint: disable=import-error

from pytorch_boundaries.utils.model_utils import crop_tensor, get_upsampling_weight


def conv_weights_init(m):
  if isinstance(m, nn.Conv2d):
    # m.weight.data.normal_(0, 0.1)
    m.weight.data.zero_()
    if m.bias is not None:
        # m.bias.data.normal_(0, 0.01)
        m.bias.data.zero_()
  elif isinstance(m, nn.BatchNorm2d):
    nn.init.uniform_(m.weight, 0., 1.)
    nn.init.constant_(m.bias, 0)


class VGG_HED(nn.Module):
  def __init__(self, config):
    super(VGG_HED, self).__init__()
    self.model_name = config.model_name
    self.num_classes = config.num_classes
    self.rgb_mean = np.array((0.485, 0.456, 0.406)) * 255.
    self.rgb_std = np.array((0.229, 0.224, 0.225)) * 255.
    # Convert to n, c, h, w
    self.rgb_mean = self.rgb_mean.reshape((1, 3, 1, 1))
    self.rgb_mean = torch.Tensor(self.rgb_mean).float().cuda()
    self.rgb_std = self.rgb_std.reshape((1, 3, 1, 1))
    self.rgb_std = torch.Tensor(self.rgb_std).float().cuda()
    if self.model_name.startswith("vgg16_bn"):
      model = models.vgg16_bn(pretrained=True).cuda()
    elif self.model_name.startswith("vgg16"):
      model = models.vgg16(pretrained=True).cuda()
    # Pad input before VGG
    self.first_padding = nn.ZeroPad2d(35)

    self.conv_1 = self.extract_layer(model, 
                                     self.model_name, 
                                     1)
    self.conv_2 = self.extract_layer(model, 
                                     self.model_name, 
                                     2)
    self.conv_3 = self.extract_layer(model, 
                                     self.model_name, 
                                     3)
    self.conv_4 = self.extract_layer(model, 
                                     self.model_name, 
                                     4)
    self.conv_5 = self.extract_layer(model, 
                                     self.model_name, 
                                     5)
    self.dsn1 = nn.Conv2d(64, 1, 1)
    self.dsn2 = nn.Conv2d(128, 1, 1)
    self.dsn3 = nn.Conv2d(256, 1, 1)
    self.dsn4 = nn.Conv2d(512, 1, 1)
    self.dsn5 = nn.Conv2d(512, 1, 1)

    self.dsn2_up = nn.ConvTranspose2d(1, 1, 4,
                                      stride=2)
    self.dsn3_up = nn.ConvTranspose2d(1, 1, 8,
                                      stride=4)
    self.dsn4_up = nn.ConvTranspose2d(1, 1, 16,
                                      stride=8)
    self.dsn5_up = nn.ConvTranspose2d(1, 1, 32,
                                      stride=16)

    self.fuse_conv = nn.Conv2d(5, 1, 1)

    init_conv_layers = [self.dsn1, self.dsn2, self.dsn3, 
                        self.dsn4, self.dsn5,
                        ]

    for layer in init_conv_layers:
      layer.apply(conv_weights_init)

    self.fuse_conv.weight.data.fill_(0.2)
    self.fuse_conv.bias.data.fill_(0) 
    self.upconv_weights_init()

  def upconv_weights_init(self):
    """Initialize the transpose convolutions."""
    for name, m in self.named_modules():
      if isinstance(m, nn.ConvTranspose2d):
        m.weight.requires_grad = False
        m.bias.requires_grad = False
        dsn_idx = int(name.split("dsn")[-1].split("_")[0])
        kernel_size = 2**dsn_idx
        m.weight.copy_(get_upsampling_weight(1, 1, kernel_size))
        nn.init.constant_(m.bias, 0)
      
  def standardize(self, inputs):
    """Mean normalize input images."""
    inputs = inputs - self.rgb_mean
    inputs = inputs / self.rgb_std
    return inputs
  
  def forward(self, inputs):
    _, _, h, w = inputs.shape
    net = self.standardize(inputs)
    net = self.first_padding(net)
    net = self.conv_1(net)
    self.side_output_1 = crop_tensor(self.dsn1(net), h, w)

    net = self.conv_2(net)
    self.side_output_2 = crop_tensor(
                          self.dsn2_up(self.dsn2(net)), 
                          h, w)

    net = self.conv_3(net)
    self.side_output_3 = crop_tensor(
                          self.dsn3_up(self.dsn3(net)),
                          h, w)

    net = self.conv_4(net)
    self.side_output_4 = crop_tensor(
                          self.dsn4_up(
                              self.dsn4(net)),
                          h, w)

    net = self.conv_5(net)
    self.side_output_5 = crop_tensor(
                          self.dsn5_up(
                              self.dsn5(net)),
                          h, w)

    stacked_outputs = torch.cat((self.side_output_1,
                                 self.side_output_2,
                                 self.side_output_3,
                                 self.side_output_4,
                                 self.side_output_5,),
                                 dim=1)
    net = self.fuse_conv(stacked_outputs)
    return_dict = {"fused_output": net,
                   "side_output_1": self.side_output_1,
                   "side_output_2": self.side_output_2,
                   "side_output_3": self.side_output_3,
                   "side_output_4": self.side_output_4,
                   "side_output_5": self.side_output_5,
                   }
    return return_dict
      
  def extract_layer(self, model, backbone_mode, ind):
    if backbone_mode=='vgg16':
        index_dict = {
            1: (0,4), 
            2: (4,9), 
            3: (9,16), 
            4: (16,23),
            5: (23,30)}
    elif backbone_mode=='vgg16_bn':
        index_dict = {
            1: (0,6), 
            2: (6,13), 
            3: (13,23), 
            4: (23,33),
            5: (33,43) }

    start, end = index_dict[ind]
    modified_model = nn.Sequential(*list(
      model.features.children()
      )[start:end])
    return modified_model