"""Schwartz and Simoncelli 2001 + excitation and inhibition, in pytorch."""
import torch  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import torch.nn.functional as F  # pylint: disable=import-error
import numpy as np
import math

def genGabor(sz, theta, gamma, sigma, sf, phi=0, contrast=2):
    """Generate gabor filter based on argument parameters."""
    location = (sz[0] // 2, sz[1] // 2)
    [x, y] = np.meshgrid(np.arange(sz[0])-location[0],
                         np.arange(sz[1])-location[1])

    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    envelope = .5 * contrast * \
        np.exp(-(x_theta**2 + (y_theta * gamma)**2)/(2 * sigma**2))
    gabor = envelope * np.cos(2 * math.pi * x_theta * sf + phi)
    return gabor

def generate_gabor_filter_weights(sz, l_theta, l_sfs,
                                  l_phase, gamma=1,
                                  contrast=1, return_dict=False):
    """Generate a bank of gabor filter weights.
    Args:
      sz: (filter height, filter width), +-2 SD of gaussian envelope
      l_theta: List of gabor orientations
      l_sfs: List of spatial frequencies, cycles per SD of envelope
      l_phase: List of gabor phase
    Returns:
      gabor filter weights with parameters sz X l_theta X l_sfs X l_phase
    """
    gabor_bank = []
    theta2filter = {}
    for theta in l_theta:
        curr_filters = []
        for sf in l_sfs:
            for phase in l_phase:
                g = genGabor(sz=(sz, sz), theta=theta,
                             gamma=gamma, sigma=sz/4,
                             sf=sf/sz, phi=phase,
                             contrast=contrast)
                gabor_bank.append(g)
                curr_filters.append(g)
        theta2filter[theta] = torch.from_numpy(
            np.array(curr_filters, dtype=np.float32))
    theta2filter = {t: torch.unsqueeze(g_b, 1)
                    for t, g_b in theta2filter.items()}
    gabor_bank = np.array(gabor_bank, dtype=np.float32)
    gabor_bank = np.expand_dims(gabor_bank, 1)
    if return_dict:
        return gabor_bank, theta2filter
    return gabor_bank


# define normalized 2D gaussian
def get_gaussian_filterbank(n_filters, f_sz, device='cuda'):
    """Generate a torch tensor for conv2D weights resembling 2D gaussian"""
    def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
        return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))

    def normalize(x):
        mid_x, mid_y = x.shape[0] // 2, x.shape[1] // 2
        x = x / x[mid_x][mid_y]
        return x
    filters = []
    x = np.linspace(-1, 1, num=f_sz)
    y = np.linspace(-1, 1, num=f_sz)
    x, y = np.meshgrid(x, y) # get 2D variables instead of 1D
    z_narrow = normalize(gaus2d(x, y, sx=.5, sy=.5))
    z_low = normalize(gaus2d(x, y, sx=1., sy=1.))
    z_mid = normalize(gaus2d(x, y, sx=2, sy=2))
    z_wide = normalize(gaus2d(x, y, sx=3, sy=3))
    filters = [z_narrow] * (n_filters // 4) + [z_low] * (n_filters // 4) + [z_mid] * (n_filters // 4) + [z_wide] * (n_filters // 4)
    filters = np.array(filters)
    filters = np.random.permutation(filters)
    filters = filters.reshape((n_filters, 1, f_sz, f_sz))
    filters = torch.Tensor(filters).float().to(device)
    return filters


def nonnegative_weights_init(m):
    """Non-negative initialization of weights."""
    if isinstance(m, nn.Conv2d):
        nn.init.uniform_(m.weight)
    else:
        m.data.fill_(0.1)
        
def orthogonal_weights_init(m):
    """Orthogonal initialization of weights."""
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight)
        m.weight.data.clamp_(0)
        m.bias.data.fill_(0.)
    else:
        m.data.fill_(0.)

def gaussian_weights_init(m):
    """Initialize weights using 2d Gaussian."""
    if isinstance(m, nn.Conv2d):
        n_filters, _, f_sz, _ = m.weight.shape
        weights = get_gaussian_filterbank(n_filters, f_sz)
        m.weight.data = weights

class GaborFilterBank(nn.Module):
    """Implements linear filtering using a Gabor Filter Bank."""

    def __init__(self,
                 in_channels,
                 l_filter_size,
                 l_theta,
                 l_sfs,
                 l_phase,
                 padding_mode='zeros',
                 contrast=1.,
                 stride=1,
                 device='cuda',
                 ):
        super(GaborFilterBank, self).__init__()
        self.l_filter_size = [int(i) for i in l_filter_size]
        self.l_theta = l_theta
        self.l_sfs = l_sfs
        self.l_phase = l_phase
        self.contrast = contrast
        self.gabor_convs = []
        self.out_dim = len(l_filter_size) * len(l_theta) * \
            len(l_sfs) * len(l_phase)

        for _, sz in enumerate(self.l_filter_size):
            filter_weights = generate_gabor_filter_weights(sz, self.l_theta,
                                                           self.l_sfs,
                                                           self.l_phase,
                                                           contrast=self.contrast)
            #print(filter_weights.shape)
            curr_conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=filter_weights.shape[0],
                                  kernel_size=sz, stride=stride,
                                  padding=(sz - 1) // 2,
                                  padding_mode=padding_mode,
                                  bias=True).to(device)
            with torch.no_grad():
                curr_conv.weight.copy_(
                    torch.from_numpy(filter_weights).float())
                curr_conv.weight.requires_grad = False
            self.gabor_convs.append(curr_conv)

    def forward(self, x):
        outputs = []
        for conv in self.gabor_convs:
            outputs.append(conv(x))
        outputs = torch.cat(outputs, 1)
        return outputs

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
                 l_theta, 
                 l_sfs,
                 l_phase,
                 divnorm_fsize=5,
                 exc_fsize=9,
                 inh_fsize=5,
                 stride=4,
                 padding_mode='zeros',
                 groups=1,
                 device='cuda',
                 alexnet_lrn=False,
                 ):
        super(DivNormExcInh, self).__init__()
        self.in_channels = in_channels
        self.alexnet_lrn = alexnet_lrn
        if in_channels <= 3:
            self.gfb = GaborFilterBank(in_channels, l_filter_size,
                                       l_theta, l_sfs, l_phase, stride=stride,
                                       padding_mode=padding_mode,
                                       contrast=1.).to(device)
            self.hidden_dim = self.gfb.out_dim
        
        if self.alexnet_lrn is False:
            self.hidden_dim = in_channels

            self.div = nn.Conv2d(
                self.hidden_dim,
                self.hidden_dim,
                divnorm_fsize,
                padding=(divnorm_fsize - 1) // 2,
                padding_mode=padding_mode,
                groups=self.hidden_dim,
                bias=False)
            gaussian_weights_init(self.div)
            self.e_e = nn.Conv2d(
                self.hidden_dim, self.hidden_dim, 
                exc_fsize, bias=True, padding=(exc_fsize - 1) // 2,
                )
            self.i_e = nn.Conv2d(
                self.hidden_dim, self.hidden_dim, inh_fsize, 
                padding=(inh_fsize - 1) // 2,
                bias=True)
            self.sigma = nn.Parameter(torch.ones([1, self.hidden_dim, 1, 1]))
            self.output_bn = nn.BatchNorm2d(in_channels)
            self.output_relu = nn.ReLU(inplace=True)
    
    def forward(self, x, residual=False, square_act=True):
        """
        params:
            x: Input activation tensor
        returns:
            output: Output post normalization
        """
        if self.alexnet_lrn:
            del residual, square_act
            return self.forward_alexnet_lrn(x)
        else:
            return self.forward_divnormei(x, residual, square_act)
        
    def forward_divnormei(self, x, residual=False, square_act=True, hor_conn=True):
        """
        params:
          x: Input grayscale image tensor
        Returns:
          output: Output post divisive normalization
        """
        identity = x
        # Gabor filter bank]
        if self.in_channels <= 3:
            simple_cells = F.relu(self.gfb(x))
            print("| Using Gabor Filter Bank |")
        else:
            simple_cells = nn.Identity()(x)
            simple_cells = simple_cells.float()
        if square_act:
            simple_cells = simple_cells ** 2
            norm = self.div(simple_cells) + self.sigma ** 2 + 1e-5
            if (norm == 0).any():
                import ipdb; ipdb.set_trace()
            simple_cells = simple_cells / norm
        else:
            norm = 1 + F.relu(self.div(simple_cells))
            simple_cells = simple_cells / norm
        # Inhibitory cells (subtractive)
        if hor_conn:
            inhibition = self.i_e(simple_cells)  # + self.i_ff(x)
            # Excitatory lateral connections (Center corresponds to self-excitation)
            excitation = self.e_e(simple_cells)
            output = self.output_bn(simple_cells + excitation - inhibition)
        else:
            output = self.output_bn(simple_cells)
        if residual:
            output += identity
        output = self.output_relu(output)
        return output

    
    def forward_lrn_alexnet(self, a):
        """
        params:
            a: Input activation (or image) tensor
        Returns:
            output: Output post local response normalization 
            as described in https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
        """
        pass