"""Common model utilities."""

import numpy as np
import torch  # pylint: disable=import-error
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