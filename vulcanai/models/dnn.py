# coding=utf-8
"""Defines the DenseNet class."""
import torch
import torch.nn as nn

from .basenetwork import BaseNetwork
from .layers import DenseUnit, FlattenUnit

import logging
from inspect import getfullargspec

from collections import OrderedDict

logger = logging.getLogger(__name__)


class DenseNetConfig:
    """Defines the necessary configuration for a DenseNet."""

    def __init__(self, raw_config):
        """
        Take in user config dict and clean it up.

        Cleaned units is stored in self.units

        Parameters:
            raw_config : dict of dict
                User specified dict

        """
        if 'dense_units' not in raw_config:
            raise KeyError("dense_units must be specified.")

        if not isinstance(raw_config['dense_units'], list):
            raise ValueError("dense_units must be of type list.")

        dense_unit_arg_spec = getfullargspec(DenseUnit)
        dense_unit_arg_spec.args.remove('self')

        # Find the index for where the defaulted values begin
        default_arg_start_index = len(dense_unit_arg_spec.args) - \
            len(dense_unit_arg_spec.defaults)
        default_args = dense_unit_arg_spec.args[default_arg_start_index:]

        # Only look at args that were specified to be overwritten
        override_args = set(default_args).intersection(set(raw_config.keys()))
        # If arguments are specified in lists,
        # check that length corresponds to dense_units
        for arg in override_args:
            if isinstance(raw_config[arg], list):
                if len(raw_config[arg]) != len(raw_config['dense_units']):
                    raise ValueError(
                        "{} list not same length as dense_units.".format(arg))
            else:
                # If a single value is specified, apply to all layers
                raw_config[arg] = [raw_config[arg]] * \
                    len(raw_config['dense_units'])

        # TODO: Think about moving dimension to config file
        _units_per_layer = list([None] + raw_config['dense_units'])
        _unit_pairs = list(zip(_units_per_layer[:-1], _units_per_layer[1:]))

        self.units = []
        for i, (in_feature, out_feature) in enumerate(_unit_pairs):
            temp_unit = {
                'in_features': in_feature,
                'out_features': out_feature
            }
            for arg in override_args:
                temp_unit[arg] = raw_config[arg][i]
            self.units.append(temp_unit)


# noinspection PyDefaultArgument,PyTypeChecker
class DenseNet(BaseNetwork):
    """
    Subclass of BaseNetwork defining a DenseNet.

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
        network : DenseNet
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
        """Define the DenseNet object."""
        super(DenseNet, self).__init__(
            name, DenseNetConfig(config), in_dim, save_path, input_networks,
            num_classes, activation, pred_activation, optim_spec,
            lr_scheduler, early_stopping, early_stopping_patience,
            early_stopping_metric,
            criter_spec, device)

    def _create_network(self, **kwargs):
        """
        Build the layers of the network into a nn.Sequential object.

        Parameters:
            dense_hid_layers : DenseNetConfig.units (list of dict)
                The hidden layers specification
            activation : torch.nn.Module
                the non-linear activation to apply to each layer

        Returns:
            output : torch.nn.Sequential
                the dense network as a nn.Sequential object

        """
        dense_hid_layers = self._config.units

        if self.input_networks:
            self.in_dim = self._get_in_dim()

        # Build network
        # Specify incoming feature size for the first dense hidden layer
        dense_hid_layers[0]['in_features'] = self.in_dim[0]
        dense_layers = OrderedDict()
        for idx, dense_layer_config in enumerate(dense_hid_layers):
            dense_layer_config['activation'] = kwargs['activation']
            layer_name = 'dense_{}'.format(idx)
            dense_layers[layer_name] = DenseUnit(**dense_layer_config)
        self.network = nn.Sequential(dense_layers)

        if self.num_classes:
            self.network.add_module(
                'classify', DenseUnit(
                    in_features=self._get_out_dim()[0],
                    out_features=self.num_classes,
                    activation=kwargs['pred_activation']))

    def _merge_input_network_outputs(self, tensors):
        output_tensors = [FlattenUnit()(t) for t in tensors]
        return torch.cat(output_tensors, dim=1)

    def __str__(self):
        if self.optim is not None:
            return super(DenseNet, self).__str__() + '\noptim: {}'\
                .format(self.optim)
        else:
            return super(DenseNet, self).__str__()
