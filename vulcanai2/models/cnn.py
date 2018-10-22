# coding=utf-8
"""Defines the ConvNet class."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .basenetwork import BaseNetwork
from .layers import DenseUnit, ConvUnit
from .utils import cast_dim

import logging
from inspect import getfullargspec
from math import sqrt, floor, ceil

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

        if len(self.in_dim) > 1 and len(self.input_networks) > 1:
            # Calculate converged in_dim for the MultiInput ConvNet
            # The new dimension to cast dense net into
            dim_tmp = []
            # Create empty input tensors
            in_tensors = [torch.zeros(x) for x in self.in_dim]
            # Calculate the number of elements in each input tensor 
            el_sizes = [t.numel() for t in in_tensors]
            #Calculate the size of each input tensor
            dim_sizes = [len(t.shape) for t in in_tensors]
            # Get max elements from all input networks
            max_el = max(el_sizes)
            max_el_ind = [i for i, j in enumerate(el_sizes) if j == max_el]
            # Get max dim from all input networks
            max_dim = max(dim_sizes)
            max_dim_ind = [i for i, j in enumerate(dim_sizes) if j == max_dim]
            
            #TODO: What if we get None? meaning, if there is no intersection
            # The ideal case would if dense_input.numel() > any of Conv input.numel()
            tensor_ind_size_ref = list(set(max_el_ind).intersection(set(max_dim_ind)))
            tensor_ind_size_ref = tensor_ind_size_ref[0]
            # TODO: Sort by number of dimensions and then by number of elements
            # to determine what to cast all the other tensors to.
            
            for t in in_tensors:

                if len(t.shape) == 1:
                    # Cast Dense to Conv1D
                    # Dense = [L] -> Conv1D = [C_{in}, L]
                    t = t.unsqueeze(0)

                if len(t.shape) > 2:
                    t = self.same_padding(t, in_tensors[tensor_ind_size_ref].shape)
                elif len(t.shape) == 2: 
                    t = t.unsqueeze(1) # Conv1D = [C_{in}, L] -> Conv2D = [C_{in}, H, W]
                    # Cast incoming dense to spatial dimensions of max size input Conv
                    n_channels = ceil(t[-1, ].numel() / in_tensors[tensor_ind_size_ref][-1, ].numel())
                    how_much_to_pad = (in_tensors[tensor_ind_size_ref][-1:].numel() * n_channels) - in_tensors[1].shape[0]
                    t = torch.cat([t, torch.zeros(*(t.shape[:-1]), how_much_to_pad)], dim = 2)
                    t = t.view(-1, *in_tensors[tensor_ind_size_ref][-1, ].shape)    
                
                dim_tmp.append(t)

            self._in_dim = [torch.cat(dim_tmp, dim=0).shape]

        # Build Network
        self.network = self._build_conv_network(
            conv_hid_layers,
            kwargs['activation'])

        self.conv_flat_dim = self.get_flattened_size(self.network) # TODO: convert to list

        if self._num_classes:
            self.out_dim = self._num_classes
            self._create_classification_layer(
                self.conv_flat_dim, kwargs['pred_activation'])
    
    def same_padding(self, tensor, cast_size):
        import numpy as np
        # Ignore channels and focus on spatial dimensions
        size_diff = np.array(cast_size[1:]) - np.array(tensor.shape[1:])
        # TODO: Use tensor.expand_as?
        padding_needed = []
        for dim in reversed(size_diff):
            dim_zero_padding = ceil(dim/2)
            dim_one_padding = floor(dim/2)
            padding_needed.append(dim_zero_padding)
            padding_needed.append(dim_one_padding)
        return F.pad(tensor, padding_needed)

    
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
