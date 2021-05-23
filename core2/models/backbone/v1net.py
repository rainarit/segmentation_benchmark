"""V1Net cell."""

import numpy as np
import torch  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import torch.nn.functional as F  # pylint: disable=import-error
import torchvision.transforms as transforms  # pylint: disable=import-error

def crop_tensor(net, out_h, out_w):
  """Crop net to input height and width."""
  _, _, in_h, in_w = net.shape
  assert in_h >= out_h and in_w >= out_w
  x_offset = (in_w - out_w) // 2
  y_offset = (in_h - out_h) // 2
  if x_offset or y_offset:
    cropped_net = net[:, :, y_offset:y_offset+out_h, x_offset:x_offset+out_w]
  return cropped_net

def get_upsampling_weight(in_channels=1, out_channels=1, kernel_size=4):
  """Make a 2D bilinear kernel suitable for upsampling"""
  factor = (kernel_size + 1) // 2
  if kernel_size % 2 == 1:
    center = factor - 1
  else:
    center = factor - 0.5
  og = np.ogrid[:kernel_size, :kernel_size]
  filt = (1 - abs(og[0] - center) / factor) * \
          (1 - abs(og[1] - center) / factor)
  weight = np.zeros((in_channels, out_channels, 
                     kernel_size, kernel_size),
                    dtype=np.float32)
  weight[range(in_channels), range(out_channels), :, :] = filt
  weight = torch.from_numpy(weight).float()
  weight = weight.cuda()
  return weight

def tile_tensor(net, timesteps):
  """Tile tensor timesteps times for temporally varying input."""
  n, c, h, w = net.shape
  net_tiled = net.repeat(timesteps, 1, 1, 1, 1).view(timesteps, n, c, h, w)
  net_tiled = torch.transpose(net_tiled, 1, 0)
  return net_tiled


def genFilterBank(nTheta=8, kernel_size=15, phase=("on", "off")):
  """Generates a bank of gabor filters."""
  def norm(x):
    """Normalize input to [-1, 1]."""
    x = (x - x.min())/(x.max() - x.min())
    return 2*x-1

  def genGabor(sz, omega, theta, func=np.cos, K=np.pi):
    """Generate a single gabor filter."""
    radius = (int(sz[0]/2.0), int(sz[1]/2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0]+1), range(-radius[1], radius[1]+1))

    x1 = x * np.cos(theta) + y * np.sin(theta)
    y1 = -x * np.sin(theta) + y * np.cos(theta)
    
    gauss = omega**2 / (4*np.pi * K**2) * np.exp(- omega**2 / (8*K**2) * ( 4 * x1**2 + y1**2))
    sinusoid = func(omega * x1) * np.exp(K**2 / 2)
    gabor = gauss * sinusoid
    return gabor

  theta = np.arange(0, np.pi, np.pi/nTheta) # range of theta
  omega = np.arange(1., 1.01, 0.1) # range of omega
  params = [(t,o) for o in omega for t in theta]
  sinFilterBank = []
  cosFilterBank = []
  gaborParams = []

  for (t, o) in params:
      gaborParam = {'omega':o, 'theta':t, 'sz':(kernel_size, kernel_size)}
      cosGabor = norm(genGabor(func=np.cos, **gaborParam))
      if "on" in phase:
        cosFilterBank.append(cosGabor)
      if "off" in phase:
        cosFilterBank.append(-cosGabor)
  cosFilterBank = np.array(cosFilterBank)
  cosFilterBank = np.expand_dims(cosFilterBank, axis=1)
  return cosFilterBank


def conv_weights_init(m):
  """Initialize conv kernel weights for V1Net."""
  if isinstance(m, nn.Conv2d):
    m.weight.data.kaiming_normal_(0, 0.1)
    if m.bias is not None:
      m.bias.data.zero_()

def gabor_init(m, nTheta=8, kernel_size=15):
  if isinstance(m, nn.Conv2d):
    filters = np.float32(genFilterBank(nTheta=nTheta, 
                                       kernel_size=kernel_size))
    filters = torch.from_numpy(filters)
    m.weight.data = filters
    m.weight.requires_grad = False
    if m.bias is not None:
      m.bias.data.zero_()

def horizontal_weights_init(m):
  """Non-negative initialization of horizontal connection weights."""
  if isinstance(m, nn.Conv2d):
    m.weight.data.uniform_(0, 0.1)
    m.weight.data.clamp_(0)
    if m.bias is not None:
      m.bias.data.zero_()


class V1NetCell(nn.Module):
  def __init__(self,
               input_dim, 
               hidden_dim,
               kernel_size, 
               kernel_size_exc,
               kernel_size_inh,
               device='cuda',
               ):
    """
    Initialize V1Net cell.
    params:
      input_dim: Integer number of channels of input tensor.
      hidden_dim: Integer number of channels of hidden state.
      kernel_size: Tuple size of the convolutional kernel.
      kernel_size_exc: Integer kernel size of excitatory kernel dims
      kernel_size_inh: Integer kernel size of inhibitory kernel dims
      bias: Boolean Whether or not to add the bias.
    """

    super(V1NetCell, self).__init__()
    print("Created V1Net")
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim

    self.kernel_size = int(kernel_size)
    self.kernel_size_exc = kernel_size_exc
    self.kernel_size_inh = kernel_size_inh
    self.padding_xh = (kernel_size - 1) // 2, (kernel_size - 1) // 2
    self.padding_exc = (self.kernel_size_exc - 1) // 2, (self.kernel_size_exc - 1) // 2
    self.padding_inh = (self.kernel_size_inh - 1) //2, (self.kernel_size_inh - 1) // 2
    self.xh_depth = self.input_dim + self.hidden_dim
    self.device = device

    self.conv_xh = nn.Sequential(
                    nn.Conv2d(self.xh_depth, 
                              self.xh_depth,
                              self.kernel_size,
                              groups=self.xh_depth,
                              padding=self.padding_xh,
                              ),
                    nn.Conv2d(self.xh_depth, 3 * self.hidden_dim, 1))
    self.conv_exc = nn.Sequential(
                      nn.Conv2d(self.hidden_dim, 
                                self.hidden_dim, 
                                self.kernel_size_exc,
                                groups=self.hidden_dim,
                                padding=self.padding_exc,
                                ),
                      nn.Conv2d(self.hidden_dim, self.hidden_dim, 1))
    self.conv_inh = nn.Sequential(
                      nn.Conv2d(self.hidden_dim, 
                                self.hidden_dim,
                                self.kernel_size_inh,
                                groups=self.hidden_dim,
                                padding=self.padding_inh,
                                ),
                      nn.Conv2d(self.hidden_dim, self.hidden_dim, 1))
    conv_layers = [self.conv_xh, self.conv_exc, self.conv_inh]
    for layer in conv_layers:
      conv_weights_init(layer)
  
  def horizontal(self, x_hor, h_hor):
    """Applies horizontal convolutions.
    params:
      x_hor: Torch tensor of horizontal input.
      h_hor: Tuple of hidden excitatory and horizontal activity.
    """
    h_exc, h_shunt = h_hor
    out_hor = torch.sigmoid(h_shunt) * (x_hor + torch.sigmoid(h_exc))
    return out_hor

  def forward(self, x, hidden):
    h, c = hidden
    x_h = torch.cat([x, h], dim=1)  # concatenate along channel axis
    res_x_h = self.conv_xh(x_h)
    res_exc = self.conv_exc(h)
    res_inh = self.conv_inh(h)
    h_hor = (res_exc, res_inh)

    f_g, g_g, o_g = torch.split(res_x_h, self.hidden_dim, dim=1)
    f = torch.sigmoid(f_g)
    o = torch.sigmoid(o_g)
    x_hor = F.relu(g_g)

    g = self.horizontal(x_hor, h_hor)

    c_next = f * c + (1-f) * g
    h_next = o * F.relu(F.layer_norm(c_next, c_next.shape[1:]))

    return h_next, c_next

  def init_hidden(self, batch_size, image_size):
    height, width = image_size
    return (torch.zeros(batch_size, 
                        self.hidden_dim, 
                        height, 
                        width, 
                        device=self.device),
            torch.zeros(batch_size, 
                        self.hidden_dim, 
                        height, 
                        width, 
                        device=self.device))


class V1Net(nn.Module):
  """
  params:
    input_dim: Number of channels in input
    hidden_dim: Number of hidden channels
    kernel_size: Size of kernel in convolutions
    kernel_size_exc: kernel size for excitatory kernel dims
    kernel_size_inh: kernel size for inhibitory kernel dims
  Output:
    A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
        0 - layer_output_list is the list of lists of length T of each output
        1 - last_state_list is the list of last states
                each element of the list is a tuple (h, c) for hidden state and memory
  Example:
      >> x = torch.zeros((1, 5, 64, 100, 100)).cuda()
      >> net = V1Net(64, 64, 5, 3, 1.5)
      >> net.to('cuda')
      >> out = net(x)
  """
  def __init__(self, input_dim, hidden_dim, 
               kernel_size, kernel_size_exc,
               kernel_size_inh):
    super(V1Net, self).__init__()

    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.kernel_size = kernel_size
    self.kernel_size_exc = kernel_size_exc
    self.kernel_size_inh = kernel_size_inh
    
    self.cell = V1NetCell(input_dim=self.input_dim,
                          hidden_dim=self.hidden_dim,
                          kernel_size=self.kernel_size,
                          kernel_size_exc = self.kernel_size_exc,
                          kernel_size_inh = self.kernel_size_inh)

  def forward(self, input_tensor, hidden_state=None):
    """
    Run V1Net iterations.
    params:
      input_tensor: 5-D Tensor of shape (b, t, c, h, w)
    Returns:
      last_state_list, layer_output
    """
    b, _, _, d_h, d_w = input_tensor.size()
    h, c = self.cell.init_hidden(batch_size=b,
                                 image_size=(d_h, d_w),
                                 )
    seq_len = input_tensor.size(1)
    l_output = []
    for t in range(seq_len):
      h, c = self.cell(x=input_tensor[:, t, :, :, :],
                       hidden=[h, c])
      l_output.append(h)
    layer_output = torch.stack(l_output, dim=1)
    last_state = [h, c]
    return layer_output, last_state


class ReducedV1NetCell(nn.Module):
  def __init__(self,
               input_dim, 
               hidden_dim,
               kernel_size=1, 
               kernel_size_exc=1,
               kernel_size_inh=1,
               device='cuda',
               ):
    """
    Initialize V1Net cell.
    params:
      input_dim: Integer number of channels of input tensor.
      hidden_dim: Integer number of channels of hidden state.
      kernel_size: Tuple size of the convolutional kernel.
      kernel_size_exc: Integer kernel size of excitatory kernel dims
      kernel_size_inh: Integer kernel size of inhibitory kernel dims
      bias: Boolean Whether or not to add the bias.
    """

    super(ReducedV1NetCell, self).__init__()
    print("Created V1Net")
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim

    self.kernel_size = int(kernel_size)
    self.kernel_size_exc = int(kernel_size_exc)
    self.kernel_size_inh = int(kernel_size_inh)
    self.padding_xh = (kernel_size - 1) // 2, (kernel_size - 1) // 2
    self.padding_exc = (self.kernel_size_exc - 1) // 2, (self.kernel_size_exc - 1) // 2
    self.padding_inh = (self.kernel_size_inh - 1) //2, (self.kernel_size_inh - 1) // 2
    self.device = device

    self.conv_xg = nn.Conv2d(self.input_dim, 
                             3 * self.hidden_dim,  # for the three gates
                             self.kernel_size,
                             )
    self.conv_hg = nn.Conv2d(self.hidden_dim, 
                             3 * self.hidden_dim,  # for the three gates
                             1)

    self.conv_exc = nn.Conv2d(self.hidden_dim, 
                              self.hidden_dim, 
                              self.kernel_size_exc,
                              padding=self.padding_exc,
                              )

    self.conv_inh = nn.Conv2d(self.hidden_dim, 
                              self.hidden_dim,
                              self.kernel_size_inh,
                              padding=self.padding_inh,
                              )

    conv_layers = [self.conv_xg, self.conv_hg]
    horizontal_layers = [self.conv_exc, self.conv_inh]
    for layer in conv_layers:
      conv_weights_init(layer)
    
    for layer in horizontal_layers:
      horizontal_weights_init(layer)
  
  def horizontal(self, x_hor, h_hor):
    """Applies horizontal convolutions.
    params:
      x_hor: Torch tensor of horizontal input.
      h_hor: Tuple of hidden excitatory and horizontal activity.
    """
    h_exc, h_shunt = h_hor
    out_hor = torch.sigmoid(-h_shunt) * (x_hor +  h_exc)
    return out_hor

  def forward(self, x, hidden):
    h, c = hidden
    res_xh = torch.sigmoid(
        self.conv_xg(x) + self.conv_hg(h))
    i, f, o = torch.split(res_xh, self.hidden_dim, dim=1)

    res_exc = self.conv_exc(h)
    res_inh = self.conv_inh(h)
    h_hor = (res_exc, res_inh)

    c_next = f * c + i * self.horizontal(h, h_hor)
    h_next = o * torch.tanh(F.layer_norm(c_next, c_next.shape[1:]))
    return h_next, c_next

  def init_hidden(self, batch_size, image_size):
    height, width = image_size
    return (torch.zeros(batch_size, 
                        self.hidden_dim, 
                        height, 
                        width, 
                        device=self.device),
            torch.zeros(batch_size, 
                        self.hidden_dim, 
                        height, 
                        width, 
                        device=self.device))


class ReducedV1Net(nn.Module):
  """
  params:
    input_dim: Number of channels in input
    hidden_dim: Number of hidden channels
    kernel_size: Size of kernel in convolutions
    kernel_size_exc: kernel size for excitatory kernel dims
    kernel_size_inh: kernel size for inhibitory kernel dims
  Output:
    A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
        0 - layer_output_list is the list of lists of length T of each output
        1 - last_state_list is the list of last states
                each element of the list is a tuple (h, c) for hidden state and memory
  Example:
      >> x = torch.zeros((1, 5, 64, 100, 100)).cuda()
      >> net = V1Net(64, 64, 5, 3, 1.5)
      >> net.to('cuda')
      >> out = net(x)
  """
  def __init__(self, input_dim, hidden_dim, 
               kernel_size, kernel_size_exc,
               kernel_size_inh, timesteps):
    super(ReducedV1Net, self).__init__()

    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.kernel_size = kernel_size
    self.kernel_size_exc = kernel_size_exc
    self.kernel_size_inh = kernel_size_inh
    self.timesteps = timesteps

    self.input_conv = nn.Conv2d(self.input_dim, 
                                self.hidden_dim, 
                                self.kernel_size,
                                padding=((self.kernel_size-1) // 2,
                                         (self.kernel_size-1) // 2,))
    gabor_init(self.input_conv, 
               nTheta=self.hidden_dim/2, 
               kernel_size=self.kernel_size)
    
    self.cell = ReducedV1NetCell(input_dim=self.input_dim,
                                 hidden_dim=self.hidden_dim,
                                 kernel_size=1,
                                 kernel_size_exc = self.kernel_size_exc,
                                 kernel_size_inh = self.kernel_size_inh)

  def forward(self, input_tensor, hidden_state=None):
    """
    Run V1Net iterations.
    params:
      input_tensor: 5-D Tensor of shape (b, t, c, h, w)
    Returns:
      last_state_list, layer_output
    """
    b, _, d_h, d_w = input_tensor.size()
    _, c = self.cell.init_hidden(batch_size=b,
                                 image_size=(d_h, d_w),
                                 )
    h = self.input_conv(input_tensor)
    l_output = []
    for _ in range(self.timesteps):
      h, c = self.cell(x=input_tensor,
                       hidden=[h, c])
      l_output.append(h)
    layer_output = torch.stack(l_output, dim=1)
    last_state = [h, c]
    return layer_output, last_state