# coding=utf-8
"""Defines the ConvNet class."""
import torch
import torch.nn as nn

from .basenetwork import BaseNetwork
from .layers import DenseUnit, ConvUnit
from .utils import expand_dim

import logging
from inspect import getfullargspec
from math import sqrt, floor

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

    def __init__(self, name, in_dim, config, save_path=None,
                 input_networks=None, num_classes=None,
                 activation=nn.ReLU(), pred_activation=None,
                 optim_spec={'name': 'Adam', 'lr': 0.001},
                 lr_scheduler=None, early_stopping=None,
                 criter_spec=nn.CrossEntropyLoss()):
        """Define the ConvNet object."""
        nn.Module.__init__(self)
        super(ConvNet, self).__init__(
            name, in_dim, ConvNetConfig(config), save_path, input_networks,
            num_classes, activation, pred_activation, optim_spec,
            lr_scheduler, early_stopping, criter_spec)

    def _create_network(self, **kwargs):
        self._in_dim = self.in_dim
        conv_hid_layers = self._config.units

        if len(self.in_dim) > 1:
            # Calculate converged in_dim for the MultiInput ConvNet
            dim_tmp = []
            # Create empty input tensors
            in_tensors = [torch.empty(x) for x in self.in_dim]
            # Calculate the number of elements in each input tensor 
            els = [t.numel() for t in in_tensors]
            #Calculate the size of each input tensor
            dim_sizes = [len(t.size()) for t in in_tensors]

            max_el_ind = els.index(max(els))
            max_dim_ind = dim_sizes.index(max(dim_sizes))

            if max_el_ind == max_dim_ind:
                for i in range(len(in_tensors)):
                    t = in_tensors[i]
                    t = expand_dim(t, len(in_tensors[max_dim_ind].size()))
                    try:
                        t = t.view(in_tensors[max_dim_ind].size())
                    except:
                        t_tmp = torch.zeros(1, *in_tensors[max_dim_ind].size()[1:])
                        if len(in_tensors[max_dim_ind].size()) == 3:
                            k_size = floor(sqrt((t.numel())))
                            t = t.view((1, k_size, k_size))
                            t_tmp[0, :t.size()[1], :t.size()[2]] = t
                        if len(in_tensors[max_dim_ind].size()) == 4:
                            NotImplementedError
                        t = t_tmp             
                    dim_tmp.append(t)
            else:
                NotImplementedError
            self._in_dim = [torch.cat(dim_tmp, dim=0).size()]

        # Build Network
        self.network = self._build_conv_network(
            conv_hid_layers,
            kwargs['activation'])

        self.conv_flat_dim = self.get_flattened_size(self.network) # TODO: convert to list

        if self._num_classes:
            self.out_dim = self._num_classes
            self._create_classification_layer(
                self.conv_flat_dim, kwargs['pred_activation'])
    
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
        conv_hid_layers[0]['in_channels'] = self._in_dim[0][0]
        conv_layers = []
        for conv_layer_config in conv_hid_layers:
            conv_layer_config['activation'] = activation
            conv_layers.append(ConvUnit(**conv_layer_config))
        conv_network = nn.Sequential(*conv_layers)
        return conv_network

    def _create_classification_layer(self, dim, pred_activation):
        self.network_tail = DenseUnit(
            in_features=dim,
            out_features=self.out_dim,
            activation=pred_activation)

    def _forward(self, xs, **kwargs):
        """
        Computation for the forward pass of the ConvNet module.

        Parameters
        ----------
        xs : list(torch.Tensor)
            List of input tensors to pass through self.

        Returns
        -------
        output : torch.Tensor

        """
        out = []
        for x in xs:
            
            out.append(x)

        network_output = self.network(torch.cat(out, dim=1))
        return network_output
   
    def get_flattened_size(self, network):
        """
        Returns the flattened output size of a Single Input ConvNet's last layer.
        :param network: The network to flatten
        :return: The flattened output size of the conv network's last layer.
        """
        with torch.no_grad():
            x = torch.ones(1, *self._in_dim[0])
            x = network(x)
            return x.numel()

    def __str__(self):
        if self.optim is not None:
            return super(ConvNet, self).__str__() + f'\noptim: {self.optim}'
        else:
            return super(ConvNet, self).__str__()
