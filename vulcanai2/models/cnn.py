# coding=utf-8
"""Defines the ConvNet class"""
import torch
import torch.nn as nn

from .basenetwork import BaseNetwork
from .layers import DenseUnit, ConvUnit

import numpy as np
import logging

logger = logging.getLogger(__name__)


# TODO: use setters to enforce types/formats/values!
# TODO: make this a base class?
# TODO: add additional constraints in the future
class ConvNetConfig:
    """
    Defines the necessary configuration for a ConvNet.
    """
    def __init__(self, mode, filters, filter_size, stride, pool):
        self.mode = mode
        self.filters = filters
        self.filter_size = filter_size
        self.stride = stride
        self.pool = pool


class ConvNet(BaseNetwork, nn.Module):
    """
    Subclass of BaseNetwork defining a ConvNet
    """

    def __init__(self, name, dimensions, config, save_path=None, input_network=None, num_classes=None,
                 activation=nn.ReLU(), pred_activation=nn.Softmax(dim=1), optim_spec={'name': 'Adam', 'lr': 0.001},
                 lr_scheduler=None, early_stopping=None, criter_spec=nn.CrossEntropyLoss()):
        
        nn.Module.__init__(self)
        super(ConvNet, self).__init__(name, dimensions, config, save_path, input_network, num_classes,
                                      activation, pred_activation, optim_spec, lr_scheduler, early_stopping, criter_spec)

    def _create_network(self, **kwargs):

        self.in_dim = self._dimensions

        if self._input_network and self._input_network.__class__.__name__ == "ConvNet":
            if self._input_network.conv_flat_dim != self.in_dim:
                self.in_dim = self.get_flattened_size(self._input_network)
            else:
                pass

        if self._input_network and self._input_network.__class__.__name__ == "DenseNet":
            if self._input_network.dims[-1] != self.in_dim:
                self.in_dim = self.dims[-1]
            else:
                pass

        self.conv_hid_layers = self._config["conv_units"]

        # Build Network
        self.network = self._build_conv_network(self.conv_hid_layers, kwargs['activation'])

        self.conv_flat_dim = self.get_flattened_size(self.network)

        if self._num_classes:
            self.out_dim = np.reshape(self._num_classes, -1).tolist()
            self._create_classification_layer(self.conv_flat_dim, kwargs['pred_activation'])
            
            if torch.cuda.is_available():
                for module in self.modules():
                    module.cuda()

    def _create_classification_layer(self, dim, pred_activation):
        self.network_tails = nn.ModuleList([DenseUnit(dim, out_d, activation=pred_activation) for out_d in self.out_dim])

    def forward(self, x):
        """
        Defines the behaviour of the network.
        If the network is defined with `num_classes` then it is assumed to be the last network
        which contains a classification layer/classifier (network tail). The data ('x')will be passed through the
        network and then through the classifier.
        If not, the input is passed through the network and returned without passing through a classification layer.
        :param x: input torch.Tensor
        :return: output torch.Tensor
        """

        if self._input_network:
            x = self._input_network(x)

        network_output = self.network(x)

        if self._num_classes:
            network_output = network_output.view(-1, self.conv_flat_dim)
            output = []
            for network_tail in self.network_tails:
                output.append(network_tail(network_output))
            # return tensor if single tail, else list of tail tensors
            if len(output) == 1:
                return output[0]
            else:
                return output
        else:
            return network_output

    # TODO: Automatically calculate padding to be the same as input shape.
    def _build_conv_network(self, conv_hid_layers, activation):
        """
        Utility function to build the layers into a nn.Sequential object.
        :param conv_hid_layers: The hidden layers specification
        :return: the conv network as a nn.Sequential object
        """
        conv_layers = []
        for conv_layer in conv_hid_layers:
            conv_layers.append(ConvUnit(
                                    conv_dim=len(conv_layer['k_size']),
                                    in_channels=conv_layer['in_ch'],         
                                    out_channels=conv_layer['out_ch'],        
                                    kernel_size=tuple(conv_layer['k_size']), 
                                    stride=conv_layer['stride'],
                                    padding=conv_layer['padding'],
                                    initializer=conv_layer["initializer"],
                                    bias_init=conv_layer["bias_init"],
                                    norm=conv_layer["norm"],
                                    activation=activation,
                                    dp=conv_layer["dropout"]))
        conv_network = nn.Sequential(*conv_layers)
        return conv_network

    def __str__(self):
        if self.optim:
            return super(ConvNet, self).__str__() + f'\noptim: {self.optim}'
        else:
            return super(ConvNet, self).__str__()
