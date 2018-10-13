# coding=utf-8
"""Defines the ConvNet class."""
import torch
import torch.nn as nn

from .basenetwork import BaseNetwork
from .layers import DenseUnit, ConvUnit

import logging
from inspect import getfullargspec

logger = logging.getLogger(__name__)


# TODO: use setters to enforce types/formats/values!
# TODO: make this a base class?
# TODO: add additional constraints in the future
class ConvNetConfig:
    """Define the necessary configuration for a ConvNet."""

    def __init__(self, raw_config):
        """
        Take in user config dict and clean it up.

        Cleaned units is stored in self.units

        Parameters
        ----------
        raw_config : dict of dict
            User specified dict

        """
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

        Infer dimension of Conv using the kernel shape.

        Parameters
        ----------
        raw_unit : dict

        Returns
        -------
        unit : dict
            Cleaned unit config.

        """
        unit = raw_unit
        for key in self.required_args:
            if key not in unit.keys():
                raise ValueError(
                    "{} needs to be specified in your config.".format(key))
        if not isinstance(unit['kernel_size'], tuple):
            unit['kernel_size'] = tuple(unit['kernel_size'])
        unit['conv_dim'] = len(unit['kernel_size'])
        return unit


class ConvNet(BaseNetwork, nn.Module):
    """Subclass of BaseNetwork defining a ConvNet."""

    def __init__(self, name, dimensions, config, save_path=None,
                 input_network=None, num_classes=None,
                 activation=nn.ReLU(), pred_activation=None,
                 optim_spec={'name': 'Adam', 'lr': 0.001},
                 lr_scheduler=None, early_stopping=None,
                 criter_spec=nn.CrossEntropyLoss()):
        """Define the ConvNet object."""
        nn.Module.__init__(self)
        super(ConvNet, self).__init__(
            name, dimensions, ConvNetConfig(config), save_path, input_network,
            num_classes, activation, pred_activation, optim_spec,
            lr_scheduler, early_stopping, criter_spec)

    def _create_network(self, **kwargs):
        
        conv_hid_layers = self._config.units
        # Build Network
        self.network = self._build_conv_network(
            conv_hid_layers,
            kwargs['activation'])

        self.conv_flat_dim = self.get_flattened_size(self.network) # TODO: convert to list

        if self._num_classes:
            self.out_dim = self._num_classes
            self._create_classification_layer(
                self.conv_flat_dim, kwargs['pred_activation'])
            
    def _create_classification_layer(self, dim, pred_activation):
        self.network_tail = DenseUnit(
            in_features=dim,
            out_features=self.out_dim,
            activation=pred_activation)

    def _forward(self, xs, **kwargs):
        """
        Define the forward behaviour of the network.

        If the network is defined with `num_classes` then it is
        assumed to be the last network which contains a
        classification layer/classifier (network tail).
        The data ('x') will be passed through the network and
        then through the classifier. If not, the input is passed
        through the network and returned without passing through
        a classification layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to pass through self.

        Returns
        -------
        output : torch.Tensor

        """
        out = []
        for x in xs:
            # TODO: NotImplemented yet, but procesing of the multiple inputs shapes before concatenation
            # Ueful tools:https://github.com/torch/torch7/blob/master/doc/maths.md
            # https://github.com/torch/nn/blob/master/doc/table.md
            out.append(x)

        network_output = self.network(torch.cat(out, dim=1))
        return network_output

    def _build_conv_network(self, conv_hid_layers, activation):
        """
        Build the layers of the network into a nn.Sequential object.

        Parameters
        ----------
        conv_hid_layers : ConvNetConfig.units (list of dict)
            The hidden layers specification
        activation : torch.nn.Module
            the non-linear activation to apply to each layer

        Returns
        -------
        output : torch.nn.Sequential
            the conv network as a nn.Sequential object

        """
        conv_layers = []
        for conv_layer_config in conv_hid_layers:
            conv_layer_config['activation'] = activation
            conv_layers.append(ConvUnit(**conv_layer_config))
        conv_network = nn.Sequential(*conv_layers)
        return conv_network
    
    def get_flattened_size(self, network):
        """
        Returns the flattened output size of a Single Input ConvNet's last layer.
        :param network: The network to flatten
        :return: The flattened output size of the conv network's last layer.
        """
        with torch.no_grad():
            x = torch.ones(1, *self.in_dim[0])
            x = network(x)
            return x.numel()

    def __str__(self):
        """Specify how to print network as string."""
        if self.optim:
            return super(ConvNet, self).__str__() + f'\noptim: {self.optim}'
        else:
            return super(ConvNet, self).__str__()
