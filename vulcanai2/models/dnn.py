import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.activation as activations
import torch.optim as optim

from .basenetwork import BaseNetwork
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
                activation=nn.Softmax(dim=1), pred_activation=nn.Softmax(dim=1), optim_spec={'name': 'Adam', 'lr': 0.001},
                lr_scheduler=None, stopping_rule='best_validation_error', criter_spec={'name': 'CrossEntropyLoss'}):
        
        nn.Module.__init__(self)
        super(DenseNet, self).__init__(name, dimensions, config, save_path, input_network, num_classes, 
                activation, pred_activation, optim_spec, lr_scheduler, stopping_rule, criter_spec)

    def _create_network(self):
        self.in_dim = self._dimensions
        self.out_dim = self._num_classes

        dims = [self.in_dim] + self._config["units"]
        self.dense_network = self.build_dense_network(dims)
        self.dense_network.add_module('Lastlayer', DenseUnit(dims[-1], self.out_dim))  # TODO: Add ClassificationUnit?

        self.init_layers(self.modules())
        if torch.cuda.is_available():
            for module in self.modules():
                module.cuda()
        
    def forward(self, x):
        '''The feedforward step'''
        return self.dense_network(x)

    def __str__(self):
        return super(DenseNet, self).__str__() + f'\noptim: {self.optim}'