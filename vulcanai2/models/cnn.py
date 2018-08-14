import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.activation as activations
import torch.optim as optim

from .BaseNetwork import BaseNetwork
from .Layers import InputUnit, DenseUnit, ConvUnit, FlattenUnit

import numpy as np

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
class CNN(BaseNetwork, nn.Module):
    def __init__(self, name, dimensions, config, save_path=None, input_network=None, num_classes=None, 
                activation=activations.Softmax(dim=1), pred_activation=activations.Softmax(dim=1), optim_spec={'name': 'Adam', 'lr': 0.001},
                lr_scheduler=None, stopping_rule='best_validation_error', criter_spec={'name': 'CrossEntropyLoss'}):
        
        nn.Module.__init__(self)
        super(CNN, self).__init__(name, dimensions, config, save_path, input_network, num_classes, 
                activation, pred_activation, optim_spec, lr_scheduler, stopping_rule, criter_spec)

    def _create_network(self):
        self.in_dim = self._dimensions
        self.out_dim = np.reshape(self._num_classes, -1).tolist()
        self.conv_hid_layers = self._config["units"][0]
        self.dense_hid_layers = self._config["units"][1]

        # Build Convolution Network
        self.conv_network = self.build_conv_network(self.conv_hid_layers)
        # Build Dense Network
        self.dense_network = self.build_dense_network(self.dense_hid_layers)
        # Build Network's Tail
        tail_in_dim = self.dense_hid_layers[-1] if len(self.dense_hid_layers) > 0 else self.conv_out_dim
        self.network_tails = nn.ModuleList([nn.Linear(tail_in_dim, out_d) for out_d in self.out_dim])

        self.init_layers(self.modules())
        if torch.cuda.is_available():
            for module in self.modules():
                module.cuda()


    def build_conv_network(self, conv_hid_layers):
        conv_dim = len(conv_hid_layers[0][2])
        conv_layers = []
        for i, layer_param in enumerate(conv_hid_layers):
            conv_layers.append(ConvUnit(
                                        conv_dim=conv_dim,
                                        in_channels=layer_param[0],         
                                        out_channels=layer_param[1],        
                                        kernel_size=tuple(layer_param[2]), 
                                        stride=layer_param[3],
                                        padding=layer_param[4],
                                        activation=self._activation))
            #conv_layers.append(net_util.get_activation_fn(self.hid_layers_activation))
            # Don't include batch norm in the first layer
            #if self.batch_norm and i != 0:
            #    conv_layers.append(nn.BatchNorm2d(hid_layer[1]))
        conv_network = nn.Sequential(*conv_layers)
        return conv_network

    def build_dense_network(self, dense_hid_layers):

        self.conv_out_dim = self.get_conv_output_size()
        dims = [self.conv_out_dim] + dense_hid_layers
        dim_pairs = list(zip(dims[:-1], dims[1:]))
        dense_layers = []
        for in_d, out_d in dim_pairs:
            dense_layers.append(DenseUnit(
                                          in_channels=in_d,
                                          out_channels=out_d))
        dense_network = nn.Sequential(*dense_layers)
        return dense_network


    def get_conv_output_size(self):
        """
        Helper function to calculate the size of the flattened 
        features after the last conv layer
        """
        with torch.no_grad():
            x = torch.ones(1, *self.in_dim)
            x = self.conv_network(x)
            return x.numel()


    def init_layers(self, layers):
        '''
        Initializes all of the layers 
        '''
        bias_init = 0.01
        for layer in layers:
            classname = layer.__class__.__name__
            if 'BatchNorm' in classname:
                torch.nn.init.uniform_(layer.weight.data)
                torch.nn.init.constant_(layer.bias.data, bias_init)
            elif 'Linear' in classname:
                torch.nn.init.xavier_uniform_(layer.weight.data)
                torch.nn.init.constant_(layer.bias.data, bias_init)
            else:
                pass

    def forward(self, x):
        '''The feedforward step'''
        if x.dim() == 3:
            x = x.permute(2, 0, 1).clone()
            x.unsqueeze_(dim=0)
        x = self.conv_network(x)
        x = x.view(-1, self.conv_out_dim)
        x = self.dense_network(x)
        output = []
        for network_tail in self.network_tails:
            output.append(network_tail(x))
        # return tensor if single tail, else list of tail tensors
        if len(output) == 1:
            return output[0]
        else:
            return output

    def __str__(self):
        return super(CNN, self).__str__() + f'\noptim: {self.optim}'