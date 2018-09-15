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

#TODO: use setters to enforce types/formats/values!
#TODO: make this a base class?
class CNNConfig():
    def __init__(self, mode, filters, filter_size, stride, pool):
        self.mode = mode
        self.filters = filters
        self.filter_size = filter_size
        self.stride = stride
        self.pool = pool

#Where config is of type CNNConfig?
class ConvNet(BaseNetwork, nn.Module):
    def __init__(self, name, dimensions, config, save_path=None, input_network=None, num_classes=None, 
                activation=nn.ReLU(), pred_activation=nn.Softmax(dim=1), optim_spec={'name': 'Adam', 'lr': 0.001},
                lr_scheduler=None, stopping_rule='best_validation_error', criter_spec={'name': 'CrossEntropyLoss'}):
        
        nn.Module.__init__(self)
        super(ConvNet, self).__init__(name, dimensions, config, save_path, input_network, num_classes, 
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


        self.conv_hid_layers = self._config["conv_units"]

        # Build Network
        self.conv_network = self.build_conv_network(self.conv_hid_layers)
        
        self.conv_flat_dim = self.get_flattened_size(self.conv_network)

        if self._num_classes:
            self.out_dim = np.reshape(self._num_classes, -1).tolist()
            self._create_classification_layer(self.conv_flat_dim)
            
            if torch.cuda.is_available():
                for module in self.modules():
                    module.cuda()

    def _create_classification_layer(self, dim):
        self.network_tails = nn.ModuleList([DenseUnit(dim, out_d) for out_d in self.out_dim])


    def forward(self, x):
        '''The feedforward step'''

        if self._input_network: 
            x = self._input_network(x)
        
        x = self.conv_network(x)

        if self._num_classes:
            x = x.view(-1, self.conv_flat_dim)
            output = []
            for network_tail in self.network_tails:
                output.append(network_tail(x))
            # return tensor if single tail, else list of tail tensors
            if len(output) == 1:
                return output[0]
            else:
                return output
        else:
            return x

    # TODO: Automatically calculate padding to be the same as input shape.
    def build_conv_network(self, conv_hid_layers):
        conv_layers = []
        for conv_layer in conv_hid_layers:
            conv_layers.append(ConvUnit(
                                    conv_dim=len(conv_layer['k_size']),
                                    in_channels=conv_layer['in_ch'],         
                                    out_channels=conv_layer['out_ch'],        
                                    kernel_size=tuple(conv_layer['k_size']), 
                                    stride=conv_layer['stride'],
                                    padding=conv_layer['padding'],
                                    activation=self._activation))
        conv_network = nn.Sequential(*conv_layers)
        return conv_network

    def __str__(self):
        if self.optim:
            return super(ConvNet, self).__str__() + f'\noptim: {self.optim}'
        else:
            return super(ConvNet, self).__str__()
