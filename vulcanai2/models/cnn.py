__author__ = 'Caitrin'
import torch.nn as nn

import torch.nn.functional as F
import torch.nn.modules.activation as activations
import torch.optim as optim
from .BaseNetwork import BaseNetwork
from .Layers import InputUnit, DenseUnit, ConvUnit, FlattenUnit
import jsonschema

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
class CNN(BaseNetwork):
    def __init__(self, name, dimensions, config, save_path=None, input_network=None, num_classes=None, 
                activation=activations.Softmax, pred_activation=activations.Softmax, optimizer=optim.Adam, 
                learning_rate=0.001, lr_scheduler=None, stopping_rule='best_validation_error', criterion=None):

        super(CNN, self).__init__(name, dimensions, config, save_path, input_network, num_classes, 
                activation, pred_activation, optimizer, 
                learning_rate, lr_scheduler, stopping_rule, criterion)
        
    def _create_network(self):

        filters=self.config.filters
        filter_size=self.config.filter_size
        stride=self.config.stride
        pool_mode=self.config.pool["mode"]
        pool_stride=self.config.pool["stride"]

        conv_dim = len(filter_size[0])
        pools = ['max', 'average_inc_pad', 'average_exc_pad']
        if not all(len(f) == conv_dim for f in filter_size):
            raise ValueError('Each tuple in filter_size {} must have a '
                             'length of {}'.format(filter_size, conv_dim))
        if not all(len(s) == conv_dim for s in stride):
            raise ValueError('Each tuple in stride {} must have a '
                             'length of {}'.format(stride, conv_dim))
        if not all(len(p) == conv_dim for p in pool_stride):
            raise ValueError('Each tuple in pool_stride {} must have a '
                             'length of {}'.format(pool_stride, conv_dim))
        if pool_mode not in pools:
            raise ValueError('{} pooling does not exist. '
                             'Please use one of: {}'.format(pool_mode, pools))

        print("Creating {} Network...".format(self.name))
        if self.input_network is None:
            print('\tInput Layer:')
            self.input_dim = self.input_dimensions[1]
            layer = layers.InputUnit(
                              in_channels=self.input_dim,
                              out_channels=self.input_dim,
                              bias=True)
            layer_name = "{}_input".format(self.name)
            self.network.add_module(layer_name, layer)
            print('\t\t{}'.format(layer))
        else:
            for l_name, l in self.input_network['network'].network.named_children():
                self.network.add_module(l_name, l)
            layer = l
            layer_name = l_name #TODO: priya why isn't this being used??
            self.input_dim = layer.out_channels

            print('Appending layer {} from {} to {}'.format(
                self.input_network['layer'],
                self.input_network['network'].name,
                self.name))

        print('\tHidden Layer:')
        for i, (f, f_size, s, p_s) in enumerate(zip(filters,
                                                    filter_size,
                                                    stride,
                                                    pool_stride)):
            layer_name = "{}_conv{}D_{}".format(
                                    self.name, conv_dim, i)
            layer = layers.ConvUnit(
                            conv_dim=conv_dim,
                            in_channels=self.input_dim,
                            out_channels=f,
                            kernel_size=f_size,
                            stride=s,
                            pool_size=p_s,
                            activation=self.nonlinearity)
            self.network.add_module(layer_name, layer)
            print('\t\t{}'.format(layer))
            self.input_dim = layer.out_channels


        if self.num_classes is not None and self.num_classes != 0:
            self.create_classification_layer()