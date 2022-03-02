"""Schwartz and Simoncelli 2001 + excitation and inhibition, in pytorch."""
from numpy.core.numeric import True_
import torch  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import torch.nn.functional as F  # pylint: disable=import-error
import numpy as np
import math

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
                 divnorm_fsize=5,
                 exc_fsize=7,
                 inh_fsize=5,
                 padding_mode='zeros',
                 groups=1,
                 device='cuda',
                 gaussian_init=False,
                 ):
        super(DivNormExcInh, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = in_channels

        self.div = nn.Conv2d(
            self.hidden_dim,
            self.hidden_dim,
            divnorm_fsize,
            padding=(divnorm_fsize - 1) // 2,
            padding_mode=padding_mode,
            groups=groups,
            bias=False)
        if gaussian_init:
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
        
    def forward(self, x, residual=True, square_act=True, hor_conn=True):
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

        if square_act:
            simple_cells = simple_cells ** 2
            norm = self.div(simple_cells) + self.sigma ** 2 + 1e-5
            if (norm == 0).any():
                import ipdb; ipdb.set_trace()
            simple_cells = simple_cells / norm
        else:
            norm = 1 + F.relu(self.div(simple_cells))
            #norm = F.relu(self.div(simple_cells)) + self.sigma ** 2 + 1e-5
            simple_cells = simple_cells / norm

        if hor_conn:
            inhibition = self.i_e(simple_cells)  # + self.i_ff(x)
            # Excitatory lateral connections (Center corresponds to self-excitation)
            excitation = self.e_e(simple_cells)
            #output = simple_cells + excitation - inhibition
            output = self.output_bn(simple_cells + excitation - inhibition)
        else:
            output = self.output_bn(simple_cells)

        if residual:
            output += identity

        output = self.output_relu(output)
<<<<<<< HEAD
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

class GaborFilters(nn.Module):
    def __init__(self, 
        in_channels, 
        n_sigmas = 3,
        n_lambdas = 4,
        n_gammas = 1,
        n_thetas = 7,
        kernel_radius=15,
        rotation_invariant=True
    ):
        super().__init__()
        self.in_channels = in_channels
        kernel_size = kernel_radius*2 + 1
        self.kernel_size = kernel_size
        self.kernel_radius = kernel_radius
        self.n_thetas = n_thetas
        self.rotation_invariant = rotation_invariant
        def make_param(in_channels, values, requires_grad=True, dtype=None):
            if dtype is None:
                dtype = 'float32'
            values = np.require(values, dtype=dtype)
            n = in_channels * len(values)
            data=torch.from_numpy(values).view(1,-1)
            data = data.repeat(in_channels, 1)
            return torch.nn.Parameter(data=data, requires_grad=requires_grad)


        # build all learnable parameters
        self.sigmas = make_param(in_channels, 2**np.arange(n_sigmas)*2)
        self.lambdas = make_param(in_channels, 2**np.arange(n_lambdas)*4.0)
        self.gammas = make_param(in_channels, np.ones(n_gammas)*0.5)
        self.psis = make_param(in_channels, np.array([0, math.pi/2.0]))

        print(len(self.sigmas))


        thetas = np.linspace(0.0, 2.0*math.pi, num=n_thetas, endpoint=False)
        thetas = torch.from_numpy(thetas).float()
        self.register_buffer('thetas', thetas)

        indices = torch.arange(kernel_size, dtype=torch.float32) -  (kernel_size - 1)/2
        self.register_buffer('indices', indices)


        # number of channels after the conv
        self._n_channels_post_conv = self.in_channels * self.sigmas.shape[1] * \
                                     self.lambdas.shape[1] * self.gammas.shape[1] * \
                                     self.psis.shape[1] * self.thetas.shape[0] 


    def make_gabor_filters(self):

        sigmas=self.sigmas
        lambdas=self.lambdas
        gammas=self.gammas
        psis=self.psis
        thetas=self.thetas
        y=self.indices
        x=self.indices

        in_channels = sigmas.shape[0]
        assert in_channels == lambdas.shape[0]
        assert in_channels == gammas.shape[0]

        kernel_size = y.shape[0], x.shape[0]



        sigmas  = sigmas.view (in_channels, sigmas.shape[1],1, 1, 1, 1, 1, 1)
        lambdas = lambdas.view(in_channels, 1, lambdas.shape[1],1, 1, 1, 1, 1)
        gammas  = gammas.view (in_channels, 1, 1, gammas.shape[1], 1, 1, 1, 1)
        psis    = psis.view (in_channels, 1, 1, 1, psis.shape[1], 1, 1, 1)

        thetas  = thetas.view(1,1, 1, 1, 1, thetas.shape[0], 1, 1)
        y       = y.view(1,1, 1, 1, 1, 1, y.shape[0], 1)
        x       = x.view(1,1, 1, 1, 1, 1, 1, x.shape[0])

        sigma_x = sigmas
        sigma_y = sigmas / gammas

        sin_t = torch.sin(thetas)
        cos_t = torch.cos(thetas)
        y_theta = -x * sin_t + y * cos_t
        x_theta =  x * cos_t + y * sin_t
        


        gb = torch.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) \
             * torch.cos(2.0 * math.pi  * x_theta / lambdas + psis)

        gb = gb.view(-1,kernel_size[0], kernel_size[1])

        return gb


    def forward(self, x):
        batch_size = x.size(0)
        sy = x.size(2)
        sx = x.size(3)  
        gb = self.make_gabor_filters()

        assert gb.shape[0] == self._n_channels_post_conv
        assert gb.shape[1] == self.kernel_size
        assert gb.shape[2] == self.kernel_size
        gb = gb.view(self._n_channels_post_conv,1,self.kernel_size,self.kernel_size)

        res = nn.functional.conv2d(input=x, weight=gb,
            padding=self.kernel_radius, groups=self.in_channels)
       
        
        if self.rotation_invariant:
            res = res.view(batch_size, self.in_channels, -1, self.n_thetas,sy, sx)
            res,_ = res.max(dim=3)

        res = res.view(batch_size, -1,sy, sx)


        return res
=======
        return output
>>>>>>> 6fd3862cb623872b312dd59731e065843011c259
