__author__ = 'Caitrin'
import torch.nn as nn
import torch.nn.functional as F
from .AbstractNetwork import AbstractNetwork
import jsonschema


class CNN(AbstractNetwork):
    """Class to generate networks and train them."""
    def __init__(self):
        pass #TODO: use super init, but with **kwargs?


    def create_network(self, config, nonlinearity):

        jsonschema.validate(config, schemas.conv_network)

        self.create_conv_network(
            filters=config.get('filters'),
            filter_size=config.get('filter_size'),
            stride=config.get('stride'),
            pool_mode=config['pool'].get('mode'),
            pool_stride=config['pool'].get('stride'),
            nonlinearity=nonlinearity
        )


    #TODO: don't think you have the final classification layer here?
    def create_conv_network(self, filters, filter_size, stride,
                            pool_mode, pool_stride, nonlinearity):
        """
        Create a convolutional network (1D, 2D, or 3D).
        Args:
            filters: list of int. number of kernels per layer
            filter_size: list of int list. size of kernels per layer
            stride: list of int list. stride of kernels
            pool_mode: string. pooling operation
            pool_stride: list of int list. down_scaling factor
            nonlinearity: string. nonlinearity to use for each layer
        Returns a conv network
        """
        conv_dim = len(filter_size[0])
        lasagne_pools = ['max', 'average_inc_pad', 'average_exc_pad']
        if not all(len(f) == conv_dim for f in filter_size):
            raise ValueError('Each tuple in filter_size {} must have a '
                             'length of {}'.format(filter_size, conv_dim))
        if not all(len(s) == conv_dim for s in stride):
            raise ValueError('Each tuple in stride {} must have a '
                             'length of {}'.format(stride, conv_dim))
        if not all(len(p) == conv_dim for p in pool_stride):
            raise ValueError('Each tuple in pool_stride {} must have a '
                             'length of {}'.format(pool_stride, conv_dim))
        if pool_mode not in lasagne_pools:
            raise ValueError('{} pooling does not exist. '
                             'Please use one of: {}'.format(pool_mode, lasagne_pools))

        print("Creating {} Network...".format(self.name))
        if self.input_network is None:
            print('\tInput Layer:')
            self.input_dim = self.input_dimensions[1]
            layer = InputUnit(
                              in_channels=self.input_dim,
                              out_channels=self.input_dim,
                              bias=True)
            layer_name = "{}_input".format(self.name)
            self.network.add_module(layer_name, layer)
            print('\t\t{}'.format(layer))
            self.layers.append(layer)
        else:
            for l_name, l in self.input_network['network'].network.named_children():
                self.network.add_module(l_name, l)
            layer = l
            layer_name = l_name
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
            layer = ConvUnit(
                            conv_dim=conv_dim,
                            in_channels=self.input_dim,
                            out_channels=f,
                            kernel_size=f_size,
                            stride=s,
                            pool_size=p_s,
                            activation=nonlinearity)
            self.network.add_module(layer_name, layer)
            self.layers.append(layer)
            print('\t\t{}'.format(layer))
            self.input_dim = layer.out_channels
