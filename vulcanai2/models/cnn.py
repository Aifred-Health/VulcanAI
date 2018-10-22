# coding=utf-8
"""Defines the ConvNet class."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .basenetwork import BaseNetwork
from .layers import DenseUnit, ConvUnit
from .utils import cast_spatial_dim_as

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
            if isinstance(unit['kernel_size'], int):
                unit['kernel_size'] = (unit['kernel_size'],)
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
            in_tensors = [torch.ones(x) for x in self.in_dim]
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
                    # Cast Dense to size [1, out_features]]
                    t = self._convert_linear_to_conv_shape(
                        tensor=t,
                        template_tensor=in_tensors[tensor_ind_size_ref])
                elif len(t.shape) > 1:
                    t = self._pad_as(
                        tensor=t,
                        template_tensor=in_tensors[tensor_ind_size_ref])

                dim_tmp.append(t)

            self._in_dim = list(torch.cat(dim_tmp, dim=0).shape)

        # Build Network
        self.network = self._build_conv_network(
            conv_hid_layers,
            kwargs['activation'])

        self.conv_flat_dim = self.get_flattened_size(self.network) # TODO: convert to list

        if self._num_classes:
            self.out_dim = self._num_classes
            self._create_classification_layer(
                self.conv_flat_dim, kwargs['pred_activation'])

    def _convert_linear_to_conv_shape(self, tensor, template_tensor):
        """Convert Linear outputs into Conv outputs."""
        n_channels = ceil(tensor.numel() / template_tensor[-1, ].numel())
        how_much_to_pad = \
            (template_tensor[-1:].numel() * n_channels) - \
            tensor.shape[0]
        tensor = F.pad(
            tensor,
            (ceil(how_much_to_pad/2), floor(how_much_to_pad/2)))
        return tensor.view(-1, *template_tensor[-1, ].shape)

    def _pad_as(self, tensor, template_tensor):
        # Expand tensor to same spatial dimensions as template tensor
        # Ex. tensor = [12, 4] | template = [16, 8, 8] will return [12, 1, 4]
        if len(tensor.shape) < len(template_tensor.shape):
            tensor = cast_spatial_dim_as(tensor, template_tensor)
        # Ignore channels and focus on spatial dimensions
        size_diff = np.array(template_tensor.shape[1:]) - np.array(tensor.shape[1:])
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
        """Specify how to print network as string."""
        if self.optim:
            return super(ConvNet, self).__str__() + f'\noptim: {self.optim}'
        else:
            return super(ConvNet, self).__str__()
