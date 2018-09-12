import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import logging
logger = logging.getLogger(__name__)

class BaseLayer(nn.Module):
    """The base class of layer
    """
    def __init__(self):

        super(BaseLayer, self).__init__()
        
        self.batch_size = None
        self.in_shape = None #[self.batch_size, *in_shape]
        self.out_shape = None #[self.batch_size, *out_shape]
        self.in_bound_layers = []
        self.out_bound_layers = []
        
        self.kernel = None              # TODO: can be changed to something else as this variable will represent the main layer for each LayerUnits
        self.norm = None

    def forward(self, input):
        return self.kernel(input)
    
    def update_batch_size(self, bs):
        self.batch_size = bs
        self.in_shape[0] = bs
        self.out_shape[0] = bs

class DenseUnit(BaseLayer):
    def __init__(self, in_channels, out_channels, weight_init=None, bias_init=0,
                 norm=None, activation=None, dp=None):
        super(DenseUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.norm = norm
        
        self.kernel = nn.Linear(self.in_channels, self.out_channels)

        if self.weight_init:
            self.weight_init(self.kernel.weight)
        nn.init.constant_(self.kernel.bias, self.bias_init)

        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm1d(out_channels)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm1d(out_channels)

        self.activation = activation

        self.dp = dp
        if self.dp is not None:
            self.dropout = nn.Dropout(self.dp)

    def forward(self, input):

        output = self.kernel(input)

        if self.norm is not None:
            output = self.bn(output)

        if self.activation is not None:
            output = self.activation(output)

        if self.dp is not None:
            output = self.dropout(output)

        return output

class ConvUnit(BaseLayer):
    def __init__(self, conv_dim, in_channels, out_channels, kernel_size=3,
                 weight_init=nn.init.xavier_uniform_, bias_init=0,
                 stride=1, padding=2, norm=None,
                 activation=None, pool_size=None):
        super(ConvUnit, self).__init__()
        if conv_dim == 1:
            self.conv_layer = nn.Conv1d
            self.batch_norm = nn.BatchNorm1d
            self.pool_layer = nn.MaxPool1d
        elif conv_dim == 2:
            self.conv_layer = nn.Conv2d
            self.batch_norm = nn.BatchNorm2d
            self.pool_layer = nn.MaxPool2d
        elif conv_dim == 3:
            self.conv_layer = nn.Conv3d
            self.batch_norm = nn.BatchNorm3d
            self.pool_layer = nn.MaxPool3d
        else:
            self.conv_layer = None
            self.batch_norm = None
            self.pool_layer = None
            ValueError("Convolution is only supported for one of the first three dimensions")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.norm = norm
        self.kernel = self.conv_layer(
                              in_channels=self.in_channels,
                              kernel_size=self.kernel_size,
                              out_channels=self.out_channels,
                              stride=stride,
                              padding=padding)

        if self.weight_init: # SRC: https://github.com/ray-project/ray/blob/b197c0c4044f66628c6672fe78581768a54d0e59/python/ray/rllib/models/pytorch/model.py
            self.weight_init(self.kernel.weight)
        nn.init.constant_(self.kernel.bias, self.bias_init)

        # TODO: Why don't we also have instance norm here too? I think this is something we can bring up to the base class
        #       since the two classes will share norms
        if self.norm:
            self.batch_norm = self.batch_norm(num_features=out_channels)
        self.activation = activation
        self.pool = None

        if pool_size is not None:
            self.pool = self.pool_layer(kernel_size=pool_size)

    def forward(self, input):
        output = self.kernel(input)

        if self.norm is not None:
            output = self.batch_norm(output)

        if self.activation is not None:
            output = self.activation(output)

        if self.pool is not None:
            output = self.pool(output)

        return output

    def get_conv_output_size(self):
        '''Helper function to calculate the size of the flattened features after the convolutional layer'''
        with torch.no_grad():
            x = torch.ones(1, *self.in_dim)
            x = self.conv_model(x)
            return x.numel()



####### TODO: Will work on these classes below later during Vulcan2 deployment
class InputUnit(BaseLayer):
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

class View(BaseLayer):
    """
    Layer to reshape the input # TODO : Testing
    """
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)



