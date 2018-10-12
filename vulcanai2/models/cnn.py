# coding=utf-8
"""Defines the ConvNet class"""
import torch
import torch.nn as nn

from .basenetwork import BaseNetwork
from .layers import DenseUnit, ConvUnit

import numpy as np
import logging
from inspect import getfullargspec

logger = logging.getLogger(__name__)


# TODO: use setters to enforce types/formats/values!
# TODO: make this a base class?
# TODO: add additional constraints in the future
class ConvNetConfig:
    """
    Defines the necessary configuration for a ConvNet.
    """
    def __init__(self, raw_config):
        if 'conv_units' not in raw_config:
            raise KeyError("conv_units must be specified")

        # Confirm all passed units conform to Unit required arguments
        conv_unit_arg_spec = getfullargspec(ConvUnit)
        conv_unit_arg_spec.args.remove('self')
        # Deal with dim inference when cleaning unit
        conv_unit_arg_spec.args.remove('conv_dim')

        # Find the index for where the defaulted values begin
        default_arg_start_index = len(conv_unit_arg_spec.args) - \
                                  len(conv_unit_arg_spec.defaults)
        self.required_args = conv_unit_arg_spec.args[:default_arg_start_index]
        self.units = []
        for u in raw_config['conv_units']:
            unit = self._clean_unit(raw_unit=u)
            self.units.append(unit)

    def _clean_unit(self, raw_unit):
        """
        Use this to catch mistakes in each user-specified unit.
        Infer dimension of Conv using the kernel shape
        """
        unit = raw_unit
        for key in self.required_args:
            if key not in unit.keys():
                raise ValueError("{} needs to be specified in your config.".format(key))
        if not isinstance(unit['kernel_size'], tuple):
            unit['kernel_size'] = tuple(unit['kernel_size'])
        unit['conv_dim'] = len(unit['kernel_size'])
        return unit

class ConvNet(BaseNetwork, nn.Module):
    """
    Subclass of BaseNetwork defining a ConvNet
    """

    def __init__(self, name, dimensions, config, save_path=None, input_network=None, num_classes=None,
                 activation=nn.ReLU(), pred_activation=None, optim_spec={'name': 'Adam', 'lr': 0.001},
                 lr_scheduler=None, early_stopping=None, criter_spec=nn.CrossEntropyLoss()):
        
        nn.Module.__init__(self)
        super(ConvNet, self).__init__(name, dimensions, ConvNetConfig(config), save_path, input_network, num_classes,
                                      activation, pred_activation, optim_spec, lr_scheduler, early_stopping, criter_spec)

    def _create_network(self, **kwargs):

        self.in_dim = self._dimensions

        if self._input_network and \
            self._input_network.__class__.__name__ == "ConvNet":

            if self._input_network.conv_flat_dim != self.in_dim:
                self.in_dim = self.get_flattened_size(self._input_network)
            else:
                pass

        if self._input_network and \
            self._input_network.__class__.__name__ == "DenseNet":

            if self._input_network.dims[-1] != self.in_dim:
                self.in_dim = self.dims[-1]
            else:
                pass

        conv_hid_layers = self._config.units
        # Build Network
        self.network = self._build_conv_network(
            conv_hid_layers,
            kwargs['activation'])

        self.conv_flat_dim = self.get_flattened_size(self.network)

        if self._num_classes:
            self.out_dim = self._num_classes
            self._create_classification_layer(
                self.conv_flat_dim, kwargs['pred_activation'])
            
            if torch.cuda.is_available():
                for module in self.modules():
                    module.cuda()

    def _create_classification_layer(self, dim, pred_activation):
        self.network_tail = DenseUnit(
            in_features=dim,
            out_features=self.out_dim,
            activation=pred_activation)

    def forward(self, x):
        """
        Defines the behaviour of the network.
        If the network is defined with `num_classes` then it is
        assumed to be the last network which contains a
        classification layer/classifier (network tail).
        The data ('x') will be passed through the network and
        then through the classifier. If not, the input is passed
        through the network and returned without passing through
        a classification layer.
        :param x: input torch.Tensor
        :return: output torch.Tensor
        """
        if self._input_network:
            x = self._input_network(x)

        network_output = self.network(x)

        if self._num_classes:
            network_output = network_output.view(-1, self.conv_flat_dim)
            class_output = self.network_tail(network_output)
            return class_output
        else:
            return network_output

    def _build_conv_network(self, conv_hid_layers, activation):
        """
        Utility function to build the layers into a nn.Sequential object.
        :param conv_hid_layers: The hidden layers specification
        :return: the conv network as a nn.Sequential object
        """
        conv_layers = []
        for conv_layer_config in conv_hid_layers:
            conv_layer_config['activation'] = activation
            conv_layers.append(ConvUnit(**conv_layer_config))
        conv_network = nn.Sequential(*conv_layers)
        return conv_network

    def __str__(self):
        if self.optim:
            return super(ConvNet, self).__str__() + f'\noptim: {self.optim}'
        else:
            return super(ConvNet, self).__str__()
