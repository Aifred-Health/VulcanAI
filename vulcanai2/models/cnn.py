# coding=utf-8
"""Defines the ConvNet class."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .basenetwork import BaseNetwork
from .layers import DenseUnit, ConvUnit, FlattenUnit
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
    """
    Subclass of BaseNetwork defining a ConvNet.
    
    """

    def __init__(self, name, config, in_dim=None, save_path=None,
                 input_networks=None, num_classes=None,
                 activation=nn.ReLU(), pred_activation=None,
                 optim_spec={'name': 'Adam', 'lr': 0.001},
                 lr_scheduler=None, early_stopping=None,
                 criter_spec=nn.CrossEntropyLoss()):
        """Define the ConvNet object."""
        nn.Module.__init__(self)
        super(ConvNet, self).__init__(
            name, ConvNetConfig(config), in_dim, save_path, input_networks,
            num_classes, activation, pred_activation, optim_spec,
            lr_scheduler, early_stopping, criter_spec)

    def _create_network(self, **kwargs):
        self._in_dim = self.in_dim
        conv_hid_layers = self._config.units

        if len(self.in_dim) > 1 and len(self.input_networks) > 1:
            # Create empty input tensors
            in_tensors = []
            for d in self.in_dim:
                # TODO: Fix Linear in_dim
                if isinstance(d, int):
                    d = tuple([d, ])
                in_tensors.append(torch.ones([1, *d]))
            output = self._merge_input_network_outputs(in_tensors)
            self._in_dim = [tuple(output.shape[1:])]

        # Build Network
        self.network = self._build_conv_network(
            conv_hid_layers,
            kwargs['activation'])

        if self._num_classes:
            # TODO: convert to list
            conv_flat_dim = self.get_flattened_size()
            self.out_dim = self._num_classes
            self._create_classification_layer(
                conv_flat_dim, kwargs['pred_activation'])

    def _get_max_incoming_resolution(self):
        # Ignoring the channels
        spatial_inputs = []
        for net in self.input_networks:
            if isinstance(net, ConvNet):
                spatial_inputs.append(list(net.out_dim[1:]))
        max_spatial_dim = len(max(spatial_inputs, key=len))

        # Fill with zeros in missing dim to compare max size later for each dim.
        for in_spatial_dim in spatial_inputs:
            while(len(in_spatial_dim) < max_spatial_dim):
                in_spatial_dim.insert(0, 0)

        # All spatial dimensions
        # Take the max size in each dimension.
        max_conv_tensor_size = np.array(spatial_inputs).transpose().max(axis=1)
        # Attach channel placeholder
        max_conv_tensor_size = np.array(max_conv_tensor_size)

        return max_conv_tensor_size

    def _merge_input_network_outputs(self, tensors):
        
        # Calculate converged in_dim for the MultiInput ConvNet
        # The new dimension to cast dense net into
        reshaped_tensors = []
        # Determine what shape to cast to without losing any information.
        max_conv_tensor_size = self._get_max_incoming_resolution()
        for t in tensors:
            if t.dim() == 2:
                # Cast Linear output to largest Conv output shape
                t = self._cast_linear_to_conv_shape(
                    tensor=t,
                    cast_shape=max_conv_tensor_size)
            elif t.dim() > 2:
                # Cast Conv output to largest Conv output shape
                t = self._pad_as(
                    tensor=t,
                    cast_shape=max_conv_tensor_size)
            reshaped_tensors.append(t)
        return torch.cat(reshaped_tensors, dim=1)

    def _cast_linear_to_conv_shape(self, tensor, cast_shape):
        """
        Convert Linear outputs into Conv outputs.

        Parameters
        ----------
        tensor : torch.Tensor
            The Linear tensor to reshape of shape [out_features]
        cast_shape : numpy.ndarray
            The Conv shape to cast linear to of shape
            [num_channels, *spatial_dimensions].

        Returns
        -------
        tensor : torch.Tensor
            Tensor of shape [num_channels, *spatial_dimensions]

        """
        # Equivalent to calculating tensor.numel() in pytorch.
        sequence_length = cast_shape.prod()
        # How many channels to spread the information into
        # Ignore batch from linear
        n_channels = ceil(tensor[-1].numel() / sequence_length)
        # How much pad to add to either sides to reshape the linear tensor
        # into cast_shape spatial dimensions.
        how_much_pad = (sequence_length * n_channels) - tensor.shape[-1]
        tensor = F.pad(tensor, (ceil(how_much_pad/2), floor(how_much_pad/2)))
        return tensor.view(-1, n_channels, *cast_shape)

    def _pad_as(self, tensor, cast_shape):
        """
        Convert Conv outputs into Conv outputs.

        Parameters
        ----------
        tensor : torch.Tensor
            The Conv tensor to reshape of shape
            [num_channels, *spatial_dimensions]
        cast_shape : numpy.ndarray
            The Conv shape to cast incoming Conv to shape
            [num_channels, *spatial_dimensions].

        Returns
        -------
        tensor : torch.Tensor
            Tensor of shape [num_channels, *spatial_dimensions]

        """
        # Expand tensor to same spatial dimensions as template tensor
        # Ex. tensor = [batch, n_channels, W] | template = [1, n_channels, W, H]
        # will return [batch, n_channels, W, H]
        # Ignore batch of incoming tensor
        if len(tensor.shape[2:]) < len(cast_shape):
            tensor = cast_spatial_dim_as(tensor, cast_shape)
        # Ignore channels and batch and focus on spatial dimensions
        dim_size_diff = cast_shape - np.array(tensor.shape[2:])
        # TODO: Use tensor.expand_as?
        padding_needed = []
        for dim in reversed(dim_size_diff):
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
        self.network_tail = nn.Sequential(
                FlattenUnit(),
                DenseUnit(
                    in_features=dim,
                    out_features=self._num_classes,
                    activation=pred_activation))

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
        if len(xs) > 1 and self.input_networks is not None:
            output = self._merge_input_network_outputs(xs)
        else:
            output = torch.cat(xs, dim=1)

        return self.network(output)

    def get_flattened_size(self):
        """
        Returns the flattened output size of a Single Input ConvNet's last layer.
        :param network: The network to flatten
        :return: The flattened output size of the conv network's last layer.
        """
        with torch.no_grad():
            x = torch.empty(1, *self._in_dim[0])
            x = self.network(x)
            x = FlattenUnit()(x)
            return x.shape[-1]

    def __str__(self):
        """Specify how to print network as string."""
        if self.optim:
            return super(ConvNet, self).__str__() + f'\noptim: {self.optim}'
        else:
            return super(ConvNet, self).__str__()
