"""Schwartz and Simoncelli 2001 + excitation and inhibition, in pytorch."""
import torch  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import torch.nn.functional as F  # pylint: disable=import-error
from early_vision.models.divisive_normalization.gabor_filter_bank import GaborFilterBank


def nonnegative_weights_init(m):
    """Non-negative initialization of weights."""
    if isinstance(m, nn.Conv2d):
        m.weight.data.uniform_(0, 1)
        m.weight.data.clamp_(0)
        if m.bias is not None:
            raise ValueError("Convolution should not contain bias")
    else:
        m.data.zero_()
        
def orthogonal_weights_init(m):
    """Orthogonal initialization of weights."""
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight)  #.data.uniform_(0, 1)
        m.weight.data.clamp_(0)
        if m.bias is not None:
            raise ValueError("Convolution should not contain bias")
    else:
        m.data.zero_()


class DivNormExcInh(nn.Module):
    """
    Implements Schwartz and Simoncelli 2001 style divisive normalization.
    params:
      input_dim: Number of channels in input
      hidden_dim: Number of hidden channels
      kernel_size: Size of kernel in convolutions
    Example:
      x = torch.zeros(1, 1, 100, 100)
      net = DivNormExcInh(1, 16, 15)
      out = net(x)
    """

    def __init__(self,
                 in_channels,
                 l_filter_size,
                 l_theta, l_sfs,
                 l_phase,
                 divnorm_fsize=5,
                 exc_fsize=9,
                 inh_fsize=5,
                 stride=4,
                 padding_mode='zeros',
                 groups=1,
                 device='cuda',
                 ):
        super(DivNormExcInh, self).__init__()
        self.in_channels = in_channels
        self.gfb = GaborFilterBank(in_channels, l_filter_size,
                                   l_theta, l_sfs, l_phase, stride=stride,
                                   padding_mode=padding_mode,
                                   contrast=1.).to(device)
        self.hidden_dim = self.gfb.out_dim
        self.div = nn.Conv2d(
            self.hidden_dim,
            self.hidden_dim,
            divnorm_fsize,
            padding=(divnorm_fsize - 1) // 2,
            padding_mode=padding_mode,
            groups=groups,
            bias=False)
        self.e_e = nn.Conv2d(
            self.hidden_dim, self.hidden_dim, 
            exc_fsize, bias=False, padding=(exc_fsize - 1) // 2,
            )
        # self.i_ff = nn.Conv2d(
        #     self.in_channels, self.hidden_dim, inh_fsize, 
        #     padding=(inh_fsize - 1) // 2, stride=stride,
        #     bias=False,)
        self.i_e = nn.Conv2d(
            self.hidden_dim, self.hidden_dim, inh_fsize, 
            padding=(inh_fsize - 1) // 2,
            bias=False,)
        # self.e_i = nn.Conv2d(self.hidden_dim, self.hidden_dim, 1, bias=False)
        self.sigma = nn.Parameter(torch.ones([1, self.hidden_dim, 1, 1]))
        nonnegative_weights_init(self.e_e)
        nonnegative_weights_init(self.i_e)
        # nonnegative_weights_init(self.e_i)
        nonnegative_weights_init(self.div)

    def forward(self, x, use_gabor=False):
        """
        params:
          x: Input grayscale image tensor
        Returns:
          output: Output post divisive normalization
        """
        # Gabor filter bank
        if use_gabor == True:
            simple_cells = F.relu(self.gfb(x))
        else:
            simple_cells = nn.Identity(x)
        # # Divisive normalization, Schwartz and Simoncelli 2001
        simple_cells = torch.pow(simple_cells, 2)
        norm = self.div(simple_cells) + self.sigma**2 + torch.tensor(1e-8)
        simple_cells = simple_cells / norm
        # Inhibitory cells (subtractive)
        inhibition = self.i_e(simple_cells)  # + self.i_ff(x)
        # Excitatory lateral connections (Center corresponds to self-excitation)
        excitation = self.e_e(simple_cells)
        simple_cells = simple_cells + excitation  - inhibition
        simple_cells = F.relu(simple_cells)
        output = {'out': simple_cells,
                  'norm': norm
                  }
        return output