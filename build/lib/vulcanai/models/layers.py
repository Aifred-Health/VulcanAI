# coding=utf-8
"""Define the ConvUnit and DenseUnit."""
import torch
import torch.nn as nn
import logging
from .utils import selu_bias_init_, selu_weight_init_
logger = logging.getLogger(__name__)


class BaseUnit(nn.Sequential):
    """
    The base class for all layers.

    Parameters:
        weight_init : torch.nn.init
            Torch initialization function for weights.
        bias_init : int or float
            A constant int or float to initialize biases with.
        norm : str
            'batch' for batch norm of 'instance' for instance norm.
        dropout : float between 0-1
            The probability of dropping out a feature during training.

    Returns:
        dense_unit : torch.nn.Sequential
            A single fully connected layer.

    """

    def __init__(self, weight_init=None, bias_init=None,
                 norm=None, dropout=None):
        """Initialize a base unit."""
        super(BaseUnit, self).__init__()

        self.weight_init = weight_init
        self.bias_init = bias_init
        self.norm = norm
        self.dropout = dropout

        self.in_shape = None  # [self.batch_size, *in_shape]
        self.out_shape = None  # [self.batch_size, *out_shape]
        self.in_bound_layers = []
        self.out_bound_layers = []

        self._kernel = None

    def _init_weights(self):
        """
        Initialize the weights.

        If self.weight_init is None, then pytorch default weight
        will be assigned to the kernel.
        """
        if self.weight_init:
            self.weight_init(self._kernel.weight)

    def _init_bias(self):
        """
        Initialize the bias.

        If self.bias_init is None, then pytorch default bias
        will be assigned to the kernel.
        """
        if self.bias_init:
            self.bias_init(self._kernel.bias)


class FlattenUnit(BaseUnit):
    """
    Layer to flatten the input.

    Returns:
        flatten_unit : torch.Sequential
            A flatten layer.

    """

    def __init__(self):
        """Initialize flatten layer."""
        super(FlattenUnit, self).__init__()

    def forward(self, x):
        """Maintain batch size but flatten all remaining dimensions."""
        return x.view(x.shape[0], -1)


# noinspection PyUnresolvedReferences
class DenseUnit(BaseUnit):
    """
    Define the DenseUnit object.

    Parameters:
        in_features : int
            The incoming feature size of a sample.
        out_features : int
            The number of hidden Linear units for this layer.
        weight_init : torch.nn.init
            Torch initialization function.
        bias_init : int or float
            A constant int or float to initialize biases with.
        norm : str
            'batch' for batch norm of 'instance' for instance norm.
        activation : torch.nn.Module
            An activation function to apply after Linear unit.
        dropout : float between 0-1
            The probability of dropping out a feature during training.

    Returns:
        dense_unit : torch.nn.Sequential
            A single fully connected layer.

    """

    def __init__(self, in_features, out_features,
                 weight_init=None, bias_init=None,
                 norm=None, activation=None, dropout=None):
        """Initialize a single DenseUnit (i.e. a dense layer)."""
        super(DenseUnit, self).__init__(weight_init, bias_init,
                                        norm, dropout)
        self.in_features = in_features
        self.out_features = out_features
        # Main layer
        self._kernel = nn.Linear(
                            in_features=self.in_features,
                            out_features=self.out_features)
        self.add_module('_kernel', self._kernel)

        # Norm
        if self.norm is not None:
            if self.norm == 'batch':
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
            if isinstance(activation, nn.SELU):
                self.weight_init = selu_weight_init_
                self.bias_init = selu_bias_init_

        # Dropout
        if self.dropout is not None:
            if isinstance(activation, nn.SELU):
                self.add_module(
                    '_dropout', nn.AlphaDropout(self.dropout))
            else:
                self.add_module(
                    '_dropout', nn.Dropout(self.dropout))
        self._init_weights()
        self._init_bias()


# TODO: Automatically calculate padding to be the same as input shape.
class ConvUnit(BaseUnit):
    """
    Define the ConvUnit object.

    Parameters:
        conv_dim : int
            1, 2, or 3 representing spatial dimensional inputs.
        in_channels : int
            The number of incoming channels.
        out_channels : int
            The number of convolution kernels for this layer.
        kernel_size : int or tuple
            The size of the 1, 2, or 3 dimensional convolution kernel.
        weight_init : torch.nn.init
            Torch initialization function.
        bias_init : int or float
            A constant int or float to initialize biases with.
        stride : int or tuple
            The stride of the 1, 2, or 3 dimensional convolution kernel.
        padding : int
            Number of zero-padding on both sides per dimension.
        norm : str
            'batch' for batch norm of 'instance' for instance norm.
        activation : torch.nn.Module
            An activation function to apply after Linear unit.
        pool_size : int
            Max pooling by a factor of pool_size in each dimension.
        dropout : float between 0-1
            The probability of dropping out a feature during training.

    Returns:
        conv_unit : torch.nn.Sequential
            A single convolution layer.

    """

    def __init__(self, conv_dim, in_channels, out_channels, kernel_size,
                 weight_init=None, bias_init=None,
                 stride=1, padding=0, norm=None,
                 activation=None, pool_size=None, dropout=None):
        """Initialize a single ConvUnit (i.e. a conv layer)."""
        super(ConvUnit, self).__init__(weight_init, bias_init,
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

        # Norm
        if self.norm:
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
            if isinstance(activation, nn.SELU):
                self.weight_init = selu_weight_init_
                self.bias_init = selu_bias_init_
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
                    '_dropout', self.dropout_layer(self.dropout))
        self._init_weights()
        self._init_bias()

    def _init_layers(self):
        if self.conv_dim == 1:
            self.conv_layer = nn.Conv1d
            self.batch_norm = nn.BatchNorm1d
            self.instance_norm = nn.InstanceNorm1d
            self.pool_layer = nn.MaxPool1d
            self.dropout_layer = nn.Dropout
        elif self.conv_dim == 2:
            self.conv_layer = nn.Conv2d
            self.batch_norm = nn.BatchNorm2d
            self.instance_norm = nn.InstanceNorm2d
            self.pool_layer = nn.MaxPool2d
            self.dropout_layer = nn.Dropout2d
        elif self.conv_dim == 3:
            self.conv_layer = nn.Conv3d
            self.batch_norm = nn.BatchNorm3d
            self.instance_norm = nn.InstanceNorm3d
            self.pool_layer = nn.MaxPool3d
            self.dropout_layer = nn.Dropout3d
        else:
            self.conv_layer = None
            self.batch_norm = None
            self.instance_norm = None
            self.pool_layer = None
            self.dropout_layer = None
            raise ValueError(
                "Convolution is only supported for"
                " one of the first three dimensions.")

    def get_conv_output_size(self):
        """Calculate the size of the flattened features after conv."""
        with torch.no_grad():
            x = torch.ones(1, *self.in_dim)
            x = self.conv_model(x)
            return x.numel()