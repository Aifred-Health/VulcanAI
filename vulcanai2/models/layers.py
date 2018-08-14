
import torch
import torch.nn as nn
import torch.nn.functional as F



class BaseLayer(nn.Module):
    """The base class of layer
    """
    def __init__(self, in_shape, out_shape, name):

        super(BaseLayer, self).__init__()
        
        self.batch_size = None
        self.in_shape = [self.batch_size, *in_shape]
        self.out_shape = [self.batch_size, *out_shape]
        self.name = name
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
        if input.dim() > 2:

            input = FlattenUnit(input.shape[1]).forward(input)

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

class FlattenUnit(BaseLayer):   # TODO: Testing
    def __init__(self, out_channels):
        super(FlattenUnit, self).__init__()
        self.out_features = out_channels

    def forward(self, input):
        input = input.contiguous().view(input.size(0), -1)
        self.flatten_layer = nn.Linear(input.size(1), self.out_features, bias=False)
        if isinstance(input, torch.cuda.FloatTensor):   # TODO: this is to ensure if the GPU is activated on the input, the flatten layer should also incorporate GPU activated
            self.flatten_layer = self.flatten_layer.cuda()
        output = self.flatten_layer(input)
        return output

    def extra_repr(self):
        return 'in_channels={}, out_features={}'.format(self.out_features,
                                                        self.out_features
                                                        )

class InputUnit(BaseLayer):
    def __init__(self, in_channels, out_channels, bias=False):
        super(InputUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = nn.Linear(self.in_channels, self.out_channels, bias=bias)

    def forward(self, input):
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


