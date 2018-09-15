import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.activation as activations
import torch.optim as optim

from .basenetwork import BaseNetwork
from .cnn import ConvNet
from .layers import DenseUnit, ConvUnit

import numpy as np
import logging
logger = logging.getLogger(__name__)

#TODO: perform typechecking

class DNNConfig():
    def __init__(self, units, dropouts):
        self.units = units
        self.dropouts = dropouts

class DenseNet(BaseNetwork, nn.Module):

    def __init__(self, name, dimensions, config, save_path=None, input_network=None, num_classes=None, 
                activation=nn.ReLU(), pred_activation=nn.Softmax(dim=1), optim_spec={'name': 'Adam', 'lr': 0.001},
                lr_scheduler=None, stopping_rule='best_validation_error', criter_spec={'name': 'CrossEntropyLoss'}):
        
        nn.Module.__init__(self)
        super(DenseNet, self).__init__(name, dimensions, config, save_path, input_network, num_classes, 
                activation, pred_activation, optim_spec, lr_scheduler, stopping_rule, criter_spec)

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
        self.dense_network = self.build_dense_network(self.dims)
        
        if self._num_classes:
            self.out_dim = self._num_classes
            self._create_classification_layer(self.dims[-1])

            if torch.cuda.is_available():
                for module in self.modules():
                    module.cuda()

    def _create_classification_layer(self, dim):
        self.network_tail = DenseUnit(dim, self.out_dim)

    def forward(self, x):
        '''The feedforward step'''
        if self._input_network: 
            x = self._input_network(x)

        if isinstance(self._input_network, ConvNet):
            x = x.view(-1, self._input_network.conv_flat_dim)

        x = self.dense_network(x)
        if self._num_classes:
            output = self.network_tail(x)
            return output
        else:
            return x

    def build_dense_network(self, dims):
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
        