# coding=utf-8
"""Defines the DenseNet class"""
import torch
import torch.nn as nn

from .basenetwork import BaseNetwork
from .cnn import ConvNet
from .layers import DenseUnit

import logging
logger = logging.getLogger(__name__)


class DenseNetConfig:
    """
    Defines the necessary configuration for a DenseNet.
    """
    def __init__(self, units, dropouts):
        self.units = units
        self.dropouts = dropouts


class DenseNet(BaseNetwork, nn.Module):
    """
    Subclass of BaseNetwork defining a DenseNet

    """

    # noinspection PyDefaultArgument
    def __init__(self, name, dimensions, config, save_path=None, input_network=None, num_classes=None,
                 activation=nn.ReLU(), pred_activation=nn.Softmax(dim=1), optim_spec={'name': 'Adam', 'lr': 0.001},
                 lr_scheduler=None, stopping_rule='best_validation_error', criter_spec={'name': 'CrossEntropyLoss'}):
        
        nn.Module.__init__(self)
        super(DenseNet, self).__init__(name, dimensions, config, save_path, input_network, num_classes,
                                       activation, pred_activation, optim_spec, lr_scheduler, stopping_rule, criter_spec
                                       )

    def _create_network(self):
        self.in_dim = self._dimensions

        if self._input_network and isinstance(self._input_network, ConvNet):
            if self._input_network.conv_flat_dim != self.in_dim:
                self.in_dim = self.get_flattened_size(self._input_network)
            else:
                pass

        if self._input_network and isinstance(self._input_network, DenseNet):
            if self._input_network.dims[-1] != self.in_dim:
                self.in_dim = self.dims[-1]
            else:
                pass

        self.dims = [self.in_dim] + self._config["dense_units"]
        self.network = self.build_dense_network(self.dims)
        
        if self._num_classes:
            self.out_dim = self._num_classes
            self._create_classification_layer(self.dims[-1])

            if torch.cuda.is_available():
                for module in self.modules():
                    module.cuda()

    def _create_classification_layer(self, dim):
        self.network_tail = DenseUnit(dim, self.out_dim)

    def forward(self, x):
        """
        Defines the behaviour of the network.
        If the network is defined with `num_classes` then it is assumed to be the last network
        which contains a classification layer/classifier (network tail). The data ('x')will be passed through the
        network and then through the classifier.
        If not, the input is passed through the network and returned without passing through a classification layer.
        :param x: the input
        :return: the output
        """
        if self._input_network:
            x = self._input_network(x)

        if isinstance(self._input_network, ConvNet):
            x = x.view(-1, self._input_network.conv_flat_dim)

        network_output = self.network(x)

        if self._num_classes:
            output = self.network_tail(network_output)
            return output
        else:
            return network_output

    def _build_dense_network(self, dims):
        # TODO: priya add types to params
        """
        Utility function to build the layers into a nn.Sequential object.
        :param dims: The dimensions
        :return: the dense network as a nn.Sequential object
        """
        dim_pairs = list(zip(dims[:-1], dims[1:]))
        dense_layers = []
        for in_d, out_d in dim_pairs:
            dense_layers.append(DenseUnit(
                                          in_features=in_d,
                                          out_features=out_d,
                                          activation=self._activation))
        dense_network = nn.Sequential(*dense_layers)
        return dense_network

    def __str__(self):
        if self.optim:
            return super(DenseNet, self).__str__() + f'\noptim: {self.optim}'
        else:
            return super(DenseNet, self).__str__()
