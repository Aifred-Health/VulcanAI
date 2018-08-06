
import torch.nn as nn
import torch.nn.functional as F


class FlattenUnit(nn.Module):
    def __init__(self, out_channels):
        super(FlattenUnit, self).__init__()
        self.out_features = out_channels

    def forward(self, input):
        input = input.view(input.size(0), -1)
        self.flatten_layer = nn.Linear(input.size(1), self.out_features, bias=False)
        output = self.flatten_layer(input)
        return output

    def extra_repr(self):
        return 'in_channels={}, out_features={}'.format(self.out_features,
                                                        self.out_features
                                                        )

class ConvUnit(nn.Module):
    def __init__(self, conv_dim, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=2, bias=True, norm=None, activation=None,
                 pool_size=None):
        super(ConvUnit, self).__init__()
        if conv_dim == 1:
            self.conv_layer = nn.Conv1d
            self.batch_norm = nn.BatchNorm1d(num_features=out_channels)
            self.pool_layer = nn.MaxPool1d
        elif conv_dim == 2:
            self.conv_layer = nn.Conv2d
            self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
            self.pool_layer = nn.MaxPool2d
        elif conv_dim == 3:
            self.conv_layer = nn.Conv3d
            self.batch_norm = nn.BatchNorm3d(num_features=out_channels)
            self.pool_layer = nn.MaxPool3d
        else:
            self.conv_layer = None
            self.batch_norm = None
            self.pool_layer = None
            ValueError("Convolution is only supported for one of the first three dimensions")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv = self.conv_layer(
                              in_channels=self.in_channels,
                              kernel_size=self.kernel_size,
                              out_channels=self.out_channels,
                              stride=stride,
                              padding=padding,
                              bias=bias)
        self.bn = norm
        self.activation = activation
        self.pool = None

        if pool_size is not None:
            self.pool = self.pool_layer(kernel_size=pool_size)

    def forward(self, input):
        output = self.conv(input)

        if self.bn is not None:
            output = self.batch_norm(output)

        if self.activation is not None:
            output = self.activation(output)

        if self.pool is not None:
            output = self.pool(output)

        return output


class DenseUnit(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, norm=None,
                 activation=None, dp=None):
        super(DenseUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.fc = nn.Linear(self.in_channels, self.out_channels, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = nn.BatchNorm1d(out_channels)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm1d(out_channels)

        self.activation = activation

        self.dp = dp
        if self.dp is not None:
            self.dropout = nn.Dropout(self.dp)

    def forward(self, input):
        if input.dim() > 2:
            input = FlattenUnit(input.shape[1]).forward(input)

        output = self.fc(input)

        if self.norm is not None:
            output = self.bn(output)

        if self.activation is not None:
            output = self.activation(output)

        if self.dp is not None:
            output = self.dropout(output)

        return output


class InputUnit(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(InputUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inp = nn.Linear(self.in_channels, self.out_channels, bias=bias)

    def forward(self, input):
        if input.dim() > 2:
            input = input.transpose(1,3) # NCHW --> NHWC
        output = self.inp(input)
        return output.transpose(1,3) # NHWC --> NCHW
