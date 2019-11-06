# coding=utf-8
"""Defines the ConvNet class."""
import torch
import torch.nn as nn
import numpy as np

from .basenetwork import BaseNetwork
from .layers import DenseUnit, ConvUnit, FlattenUnit
from .utils import pad

import logging
from inspect import getfullargspec
from math import ceil

from collections import OrderedDict

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

        Parameters:
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

        Parameters:
            raw_unit : dict

        Returns:
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


# noinspection PyDefaultArgument,PyTypeChecker
class ConvNet(BaseNetwork):
    """
    Subclass of BaseNetwork defining a ConvNet.

    Parameters:
        name : str
            The name of the network. Used when saving the file.
        config : dict
            The configuration of the network module, as a dict.
        in_dim : tuple
            The input dimensions of the network. Not required to specify when
            the network has input_networks.
        save_path : str
            The name of the file to which you would like to save this network.
        input_networks : list of BaseNetwork
            A network object provided as input.
        num_classes : int or None
            The number of classes to predict.
        activation : torch.nn.Module
            The desired activation function for use in the network.
        pred_activation : torch.nn.Module
            The desired activation function for use in the prediction layer.
        optim_spec : dict
            A dictionary of parameters for the desired optimizer.
        lr_scheduler : torch.optim.lr_scheduler
            A callable torch.optim.lr_scheduler
        early_stopping : str or None
            So far just 'best_validation_error' is implemented.
        early_stopping_patience: integer
            Number of validation iterations of decreasing loss
            (note -not necessarily every epoch!
            before early stopping is applied.
        early_stopping_metric: string
            Either "loss" or "accuracy" are implemented.
        criter_spec : dict
            criterion specification with name and all its parameters.

    Returns:
        network : ConvNet
            A network of type BaseNetwork.

    """

    def __init__(self, name, config, in_dim=None, save_path=None,
                 input_networks=None, num_classes=None,
                 activation=nn.ReLU(), pred_activation=None,
                 optim_spec={'name': 'Adam', 'lr': 0.001},
                 lr_scheduler=None, early_stopping=None,
                 early_stopping_patience=None,
                 early_stopping_metric="accuracy",
                 criter_spec=nn.CrossEntropyLoss(),
                 device="cuda:0"):
        """Define the ConvNet object."""
        super(ConvNet, self).__init__(
            name, ConvNetConfig(config), in_dim, save_path, input_networks,
            num_classes, activation, pred_activation, optim_spec,
            lr_scheduler, early_stopping, early_stopping_patience,
            early_stopping_metric,
            criter_spec, device)

    def _create_network(self, **kwargs):
        """
        Build the layers of the network into a nn.Sequential object.

        Parameters:
            conv_hid_layers : ConvNetConfig.units (list of dict)
                The hidden layers specification
            activation : torch.nn.Module
                the non-linear activation to apply to each layer

        Returns:
            output : torch.nn.Sequential
                the conv network as a nn.Sequential object

        """
        conv_hid_layers = self._config.units

        if self.input_networks:
            self.in_dim = self._get_in_dim()

        conv_hid_layers[0]['in_channels'] = self.in_dim[0]
        conv_layers = OrderedDict()
        for idx, conv_layer_config in enumerate(conv_hid_layers):
            conv_layer_config['activation'] = kwargs['activation']
            layer_name = 'conv_{}'.format(idx)
            conv_layers[layer_name] = ConvUnit(**conv_layer_config)
        self.network = nn.Sequential(conv_layers)

        if self.num_classes:
            self.network.add_module(
                'flatten', FlattenUnit())
            self.network.add_module(
                'classify', DenseUnit(
                    in_features=self._get_out_dim()[0],
                    out_features=self.num_classes,
                    activation=kwargs['pred_activation']))

    def _merge_input_network_outputs(self, tensors):
        """Calculate converged in_dim for the MultiInput ConvNet."""
        reshaped_tensors = []
        # Determine what shape to cast to without losing any information.
        max_conv_tensor_size = self._get_max_incoming_spatial_dims()
        for t in tensors:
            if t.dim() == 2:
                # Cast Linear output to largest Conv output shape
                t = self._cast_linear_to_shape(
                    tensor=t,
                    cast_shape=max_conv_tensor_size)
            elif t.dim() > 2:
                # Cast Conv output to largest Conv output shape
                t = self._cast_conv_to_shape(
                    tensor=t,
                    cast_shape=max_conv_tensor_size)
            reshaped_tensors.append(t)
        return torch.cat(reshaped_tensors, dim=1)

    def _get_max_incoming_spatial_dims(self):
        """Return the max spatial dimensions of the input networks."""
        # Ignoring the channels
        spatial_inputs = []
        for in_net in self.input_networks.values():
            if isinstance(in_net, ConvNet):
                spatial_inputs.append(list(in_net.out_dim[1:]))
        max_spatial_dim = len(max(spatial_inputs, key=len))

        # Fill with zeros in missing dim to compare
        # max size later for each dim.
        for in_spatial_dim in spatial_inputs:
            while len(in_spatial_dim) < max_spatial_dim:
                in_spatial_dim.insert(0, 0)

        # All spatial dimensions
        # Take the max size in each dimension.
        max_conv_tensor_size = np.array(spatial_inputs).transpose().max(axis=1)
        return np.array(max_conv_tensor_size)

    @staticmethod
    def _cast_linear_to_shape(tensor, cast_shape):
        """
        Convert Linear outputs into Conv outputs.

        Parameters:
            tensor : torch.Tensor
                The Linear tensor to reshape of shape [out_features]
            cast_shape : numpy.ndarray
                The Conv shape to cast linear to of shape
                [batch, num_channels, *spatial_dimensions].

        Returns:
            tensor : torch.Tensor
                Tensor of shape [batch, num_channels, *spatial_dimensions]

        """
        # Equivalent to calculating tensor.numel() in pytorch.
        sequence_length = cast_shape.prod()
        # How many channels to spread the information into
        # Ignore batch from linear
        n_channels = ceil(tensor[-1].numel() / sequence_length)
        # How much pad to add to either sides to reshape the linear tensor
        # into cast_shape spatial dimensions.
        pad_shape = sequence_length * n_channels
        tensor = pad(tensor=tensor, target_shape=[pad_shape])
        return tensor.view(-1, n_channels, *cast_shape)

    @staticmethod
    def _cast_conv_to_shape(tensor, cast_shape):
        """
        Convert Conv outputs into Conv outputs.

        Parameters:
            tensor : torch.Tensor
                The Conv tensor to reshape of shape
                [batch, num_channels, *spatial_dimensions]
            cast_shape : numpy.ndarray
                The Conv shape to cast incoming Conv to shape
                [batch, num_channels, *spatial_dimensions].

        Returns:
            tensor : torch.Tensor
                Tensor of shape [batch, num_channels, *spatial_dimensions]

        """
        # Extract only the spatial dimensions by ignoring the batch and channel
        spatial_dim_idx_start = 2
        if len(tensor.shape[spatial_dim_idx_start:]) < len(cast_shape):
            # TODO: https://github.com/pytorch/pytorch/issues/9410
            # Ignore batch for incoming tensor
            # For each missing dim, add dims until it
            # is equivalient to the max dim
            n_unsqueezes = len(cast_shape) - \
                len(tensor.shape[spatial_dim_idx_start:])
            for _ in range(n_unsqueezes):
                tensor = tensor.unsqueeze(dim=spatial_dim_idx_start)
        return pad(tensor=tensor, target_shape=cast_shape)

    def __str__(self):
        """Specify how to print network."""
        if self.optim is not None:
            return super(ConvNet, self).__str__() + '\noptim: {}'\
                .format(self.optim)
        else:
            return super(ConvNet, self).__str__()
