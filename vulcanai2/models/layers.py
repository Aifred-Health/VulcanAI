import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import logging
logger = logging.getLogger(__name__)


class BaseUnit(nn.Sequential):
    """The base class of layer
    """
    def __init__(self, weight_init=None, bias_init=0,
                 norm=None, activation=None, dp=None):

        super(BaseUnit, self).__init__()
        
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.norm = norm
        self.activation = activation
        self.dp = dp
        
        self.in_shape = None #[self.batch_size, *in_shape]
        self.out_shape = None #[self.batch_size, *out_shape]
        self.in_bound_layers = []
        self.out_bound_layers = []
        
        self.kernel = None        
        self.norm = norm
    
    def init_weights(self):
        """Initialize the weights."""
        if self.weight_init: # SRC: https://github.com/ray-project/ray/blob/b197c0c4044f66628c6672fe78581768a54d0e59/python/ray/rllib/models/pytorch/model.py
            self.weight_init(self.kernel.weight)
        nn.init.constant_(self.kernel.bias, self.bias_init)
  

class DenseUnit(BaseUnit):
    def __init__(self, in_features , out_features , weight_init=None, bias_init=0,
                 norm=None, activation=None, dp=None):
        super(DenseUnit, self).__init__(weight_init, bias_init,
                                        norm, activation, dp)
        self.in_features  = in_features 
        self.out_features  = out_features 
        
        # Main layer
        self.kernel = nn.Linear(
                            in_features=self.in_features, 
                            out_features=self.out_features
                            )
        if self.weight_init:
            self.weight_init(self.kernel.weight)
        nn.init.constant_(self.kernel.bias, self.bias_init)
        self.add_module('kernel', self.kernel)

        # Norm
        if self.norm is not None:
            if self.norm =='batch':
                self.norm = torch.nn.BatchNorm1d(self.out_features)
                self.add_module('norm', self.norm)
            elif self.norm == 'instance':
                self.norm = torch.nn.InstanceNorm1d(self.out_features)
                self.add_module('norm', self.norm)

        # Activation/Non-Linearity
        if self.activation is not None:
            self.add_module('activation', self.activation)

        # Dropout
        if self.dp is not None:
            self.dropout = nn.Dropout(self.dp)
            self.add_module('dropout', self.dropout)

        self.init_weights()

class ConvUnit(BaseUnit):
    def __init__(self, conv_dim, in_channels, out_channels, kernel_size=3,
                 weight_init=nn.init.xavier_uniform_, bias_init=0,
                 stride=1, padding=2, norm=None,
                 activation=None, pool_size=None):
        super(ConvUnit, self).__init__()
        self.conv_dim = conv_dim
        self._init_layers()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Main layer
        self.kernel = self.conv_layer(
                              in_channels=self.in_channels,
                              kernel_size=self.kernel_size,
                              out_channels=self.out_channels,
                              stride=stride,
                              padding=padding
                              )

        self.add_module('kernel', self.kernel)

        # Norm
        if self.norm is not None:
            self.norm = self.batch_norm(num_features=self.out_channels)
            self.add_module('norm', self.norm)

        # Activation/Non-Linearity
        if self.activation is not None:
            self.add_module('activation', self.activation)

        # Pool
        if pool_size is not None:
            self.pool = self.pool_layer(kernel_size=pool_size)
            self.add_module('pool', self.pool)
                    
        self.init_weights()

    def _init_layers(self):
        if self.conv_dim == 1:
            self.conv_layer = nn.Conv1d
            self.batch_norm = nn.BatchNorm1d
            self.pool_layer = nn.MaxPool1d
        elif self.conv_dim == 2:
            self.conv_layer = nn.Conv2d
            self.batch_norm = nn.BatchNorm2d
            self.pool_layer = nn.MaxPool2d
        elif self.conv_dim == 3:
            self.conv_layer = nn.Conv3d
            self.batch_norm = nn.BatchNorm3d
            self.pool_layer = nn.MaxPool3d
        else:
            self.conv_layer = None
            self.batch_norm = None
            self.pool_layer = None
            ValueError("Convolution is only supported for one of the first three dimensions")

    def get_conv_output_size(self):
        '''Helper function to calculate the size of the flattened features after the convolutional layer'''
        with torch.no_grad():
            x = torch.ones(1, *self.in_dim)
            x = self.conv_model(x)
            return x.numel()


####### TODO: Will work on these classes below later during Vulcan2 deployment
class InputUnit(BaseUnit):
    def __init__(self, in_channels, out_channels, bias=False):
        super(InputUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = nn.Linear(self.in_channels, self.out_channels, bias=bias)

    def forward(self, input):
        if input.dim() > 2:
            input = input.transpose(1,3) # NCHW --> NHWC
            output = self.kernel(input)
            return output.transpose(1,3) # NHWC --> NCHW
        else:
            output = self.kernel(input)
            return output

class View(BaseUnit):
    """
    Layer to reshape the input # TODO : Testing
    """
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)



