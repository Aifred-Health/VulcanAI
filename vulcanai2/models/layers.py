import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import logging
logger = logging.getLogger(__name__)


class BaseUnit(nn.Sequential):
    """The base class of layer
    """
    def __init__(self, initializer=None, bias_init=None,
                 norm=None, dropout=None):

        super(BaseUnit, self).__init__()
        
        self.initializer = initializer
        self.bias_init = bias_init
        self.norm = norm
        self.dropout = dropout
        
        self.in_shape = None #[self.batch_size, *in_shape]
        self.out_shape = None #[self.batch_size, *out_shape]
        self.in_bound_layers = []
        self.out_bound_layers = []
        
        self._kernel = None   
        self.norm = norm
    
    def _init_weights(self):
        """
        Initialize the weights.
        if self.initializer is None, then pytorch default weight
        will be assigned to the kernel
        """
        if self.initializer:
            self.initializer(self._kernel.weight)

    def _init_bias(self):
        """
        Initialize the bias.
        if self.bias_init is None, then pytorch default weight
        will be assigned to the kernel
        """
        if self.bias_init:
            nn.init.constant_(self._kernel.bias, self.bias_init)
  

class DenseUnit(BaseUnit):
    def __init__(self, in_features, out_features,
                 initializer=None, bias_init=None,
                 norm=None, activation=None, dropout=None):
        super(DenseUnit, self).__init__(initializer, bias_init,
                                        norm, dropout)
        self.in_features  = in_features 
        self.out_features  = out_features 
        
        # Main layer
        self._kernel = nn.Linear(
                            in_features=self.in_features, 
                            out_features=self.out_features,
                            bias=True
                            )
        self.add_module('_kernel', self._kernel)
        self._init_weights()
        self._init_bias()

        # Norm
        if self.norm is not None:
            if self.norm =='batch':
                self.add_module(
                    '_norm',
                    torch.nn.BatchNorm1d(self.out_features))
            elif self.norm == 'instance':
                self.add_module(
                    '_norm',
                    torch.nn.InstanceNorm1d(self.out_features))

        # Activation/Non-Linearity
        if activation is not None:
            self.add_module('_activation', activation)

        # Dropout
        if self.dropout is not None:
            if isinstance(activation, nn.SELU):
                self.add_module(
                    '_dropout', nn.AlphaDropout(self.dropout))
            else:
                self.add_module(
                    '_dropout', nn.Dropout(self.dropout))
   
# TODO: Automatically calculate padding to be the same as input shape.
class ConvUnit(BaseUnit):
    def __init__(self, conv_dim, in_channels, out_channels, kernel_size,
                 initializer=None, bias_init=None,
                 stride=1, padding=0, norm=None,
                 activation=None, pool_size=None, dropout=None):
        super(ConvUnit, self).__init__(initializer, bias_init,
                                        norm, dropout)
        self.conv_dim = conv_dim
        self._init_layers()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Main layer
        self._kernel = self.conv_layer(
                              in_channels=self.in_channels,
                              kernel_size=self.kernel_size,
                              out_channels=self.out_channels,
                              stride=stride,
                              padding=padding,
                              bias=True
                              )
        self.add_module('_kernel', self._kernel)
        self._init_weights()
        self._init_bias()

        # Norm
        if self.norm is not None:
            if self.norm == 'batch':
                self.add_module(
                    '_norm',
                    self.batch_norm(num_features=self.out_channels))
            elif self.norm == 'instance':
                self.add_module(
                    '_norm',
                    self.instance_norm(num_features=self.out_channels))

        # Activation/Non-Linearity
        if activation is not None:
            self.add_module('_activation', activation)

        # Pool
        if pool_size is not None:
            self.add_module(
                '_pool', self.pool_layer(kernel_size=pool_size))
        
        # Dropout
        if self.dropout is not None:
            if isinstance(activation, nn.SELU):
                self.add_module(
                    '_dropout', nn.AlphaDropout(self.dropout))
            else:
                self.add_module(
                    '_dropout', nn.Dropout(self.dropout))
          

    def _init_layers(self):
        if self.conv_dim == 1:
            self.conv_layer = nn.Conv1d
            self.batch_norm = nn.BatchNorm1d
            self.instance_norm = nn.InstanceNorm1d
            self.pool_layer = nn.MaxPool1d
        elif self.conv_dim == 2:
            self.conv_layer = nn.Conv2d
            self.batch_norm = nn.BatchNorm2d
            self.instance_norm = nn.InstanceNorm2d
            self.pool_layer = nn.MaxPool2d
        elif self.conv_dim == 3:
            self.conv_layer = nn.Conv3d
            self.batch_norm = nn.BatchNorm3d
            self.instance_norm = nn.InstanceNorm3d
            self.pool_layer = nn.MaxPool3d
        else:
            self.conv_layer = None
            self.batch_norm = None
            self.instance_norm = None
            self.pool_layer = None
            raise ValueError("Convolution is only supported"
            " for one of the first three dimensions")

    def get_conv_output_size(self):
        """
        Helper function to calculate the size of the flattened
        features after the convolutional layer
        """
        with torch.no_grad():
            x = torch.ones(1, *self.in_dim)
            x = self.conv_model(x)
            return x.numel()


####### TODO: Will work on these classes below later during
# Vulcan2 deployment
class InputUnit(BaseUnit):
    def __init__(self, in_channels, out_channels, bias=False):
        super(InputUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._kernel = nn.Linear(
            self.in_channels, self.out_channels, bias=bias)

    def forward(self, input):
        if input.dim() > 2:
            input = input.transpose(1,3) # NCHW --> NHWC
            output = self._kernel(input)
            return output.transpose(1,3) # NHWC --> NCHW
        else:
            output = self._kernel(input)
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
