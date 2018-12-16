"""Test all DenseNet capabilities."""
import pytest
import numpy as np
import copy
import pickle
import logging
import os
import shutil

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from vulcanai2.models import BaseNetwork
from vulcanai2.models.dnn import DenseNet, DenseNetConfig


class TestDenseNet:
    """Define DenseNet test class."""

    def test_init(self, dnn_noclass):
        """Initialization Test of a DenseNet object"""
        assert isinstance(dnn_noclass, BaseNetwork)
        assert isinstance(dnn_noclass, nn.Module)
        assert hasattr(dnn_noclass, 'network')
        assert hasattr(dnn_noclass, 'in_dim')
        assert hasattr(dnn_noclass, 'record')
        assert hasattr(dnn_noclass, 'device')

        assert dnn_noclass._name is not None
        assert isinstance(dnn_noclass._config, DenseNetConfig)

        assert dnn_noclass.input_networks is None
        assert dnn_noclass.epoch == 0
        assert dnn_noclass.optim == None
        assert dnn_noclass.criterion == None
        
        assert not hasattr(dnn_noclass, 'metrics')

    def test_function_multi_input(self, multi_input_dnn):
        """Test methods/functions wrt multi_input_cnn"""
        assert isinstance(multi_input_dnn.input_networks, nn.ModuleDict)
        assert len(list(multi_input_dnn.input_networks)) == 2
        assert multi_input_dnn._merge_input_network_outputs([
                                torch.ones([10, 1, 28]),
                                torch.ones([10, 1, 28, 28])
               ]).shape == (10, 812)

    def test_forward_pass_not_nan(self, dnn_noclass):
        """Confirm out is non nan."""
        test_input = torch.ones([5, *dnn_noclass.in_dim])
        test_dataloader = DataLoader(TensorDataset(test_input, test_input))
        output = dnn_noclass.forward_pass(
            data_loader=test_dataloader,
            convert_to_class=False)
        assert np.any(~np.isnan(output))

    def test_forward_pass_class_not_nan(self, dnn_class):
        """Confirm out is non nan."""
        test_input = torch.ones([5, *dnn_class.in_dim])
        test_dataloader = DataLoader(TensorDataset(test_input, test_input))
        raw_output = dnn_class.forward_pass(
            data_loader=test_dataloader,
            convert_to_class=False)
        class_output = dnn_class.metrics.extract_class_labels(
            in_matrix=test_input)
        assert np.any(~np.isnan(raw_output))
        assert np.any(~np.isnan(class_output))

    def test_freeze_class(self, dnn_class):
        """Test class network freezing."""
        dnn_class.freeze(apply_inputs=False)
        for params in dnn_class.network.parameters():
            assert params.requires_grad is False

    def test_unfreeze_class(self, dnn_class):
        """Test class network unfreezing."""
        dnn_class.freeze(apply_inputs=False)
        dnn_class.unfreeze(apply_inputs=False)
        for params in dnn_class.network.parameters():
            assert params.requires_grad is True

    def test_freeze_noclass(self, dnn_noclass):
        """Test intermediate network freezing."""
        dnn_noclass.freeze(apply_inputs=False)
        for params in dnn_noclass.network.parameters():
            assert params.requires_grad is False

    def test_unfreeze_noclass(self, dnn_noclass):
        """Test intermediate network unfreezing."""
        dnn_noclass.freeze(apply_inputs=False)
        dnn_noclass.unfreeze(apply_inputs=False)
        for params in dnn_noclass.network.parameters():
            assert params.requires_grad is True

    def test_add_input_network(self, dnn_noclass, dnn_class):
        """Test add input Network functionality."""
        dnn_class.add_input_network(dnn_noclass)
        assert dnn_class.input_networks[dnn_noclass.name] is dnn_noclass
        assert dnn_class.in_dim == dnn_noclass.out_dim
