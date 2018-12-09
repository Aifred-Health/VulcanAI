"""Test all ConvNet capabilities."""
import numpy as np
import pytest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from vulcanai2.models import BaseNetwork
from vulcanai2.models.cnn import ConvNet, ConvNetConfig
from vulcanai2.models.utils import master_device_setter


class TestConvNet:
    """Define ConvNet test class."""
    
    def test_init(self, conv1D_net):
        """Initialization Test of a ConvNet object"""
        assert isinstance(conv1D_net, BaseNetwork)
        assert isinstance(conv1D_net, nn.Module)
        assert hasattr(conv1D_net, 'network')
        assert hasattr(conv1D_net, 'in_dim')
        assert hasattr(conv1D_net, 'record')
        assert hasattr(conv1D_net, 'device')

        assert conv1D_net._name is not None
        assert isinstance(conv1D_net._config, ConvNetConfig)

        assert conv1D_net.input_networks is None
        assert conv1D_net.epoch == 0
        assert conv1D_net.optim is None
        assert conv1D_net.criterion is None

        assert not hasattr(conv1D_net, 'metrics')

    def test_mult_multi_input(self, multi_input_cnn):
        """Test methods/functions wrt multi_input_cnn."""
        assert isinstance(multi_input_cnn.input_networks, nn.ModuleDict)
        assert len(list(multi_input_cnn.input_networks)) == 3
        assert multi_input_cnn._get_max_incoming_spatial_dims() == (8, 8, 8)
        assert multi_input_cnn._merge_input_network_outputs([
                                torch.ones([10, 1, 28, 28]),
                                torch.ones([10, 1, 28, 28, 28]),
                                torch.ones(10, *multi_input_cnn.\
                                input_networks['multi_input_dnn'].out_dim)
               ]).shape == (10, 3, 8, 8, 8)

    def test_add_input_network(self, multi_input_cnn_add_input_network,
                               conv3D_net):
        """Test add input Network functionality."""
        assert isinstance(multi_input_cnn_add_input_network.input_networks,
                          nn.ModuleDict)
        assert multi_input_cnn_add_input_network\
               .input_networks[conv3D_net.name] is conv3D_net
        assert multi_input_cnn_add_input_network.in_dim == conv3D_net.out_dim
    
    def test_forward(self, conv1D_net):
        """Test Forward of ConvNet"""
        out = conv1D_net(torch.ones([10, *conv1D_net.in_dim]))
        assert out.shape == (10, 64, 1)

    def test_forward_multi_input(self, multi_input_cnn,
                                         multi_input_cnn_data):
        """Test Forward of Multi Input ConvNet"""
        master_device_setter(multi_input_cnn, 'cuda:0')
        input_tensor = [torch.ones([10, 1, 28, 28]),
               torch.ones([10, 1, 28, 28, 28]),
               [torch.ones([10, 1, 28]),
                torch.ones([10, 1, 28, 28])]
            ]
        out = multi_input_cnn(input_tensor)
        assert out.shape == (10, 10)

    def test_forward_multi_input_cnn_add_input_network(self, 
                                multi_input_cnn_add_input_network):
        """Test Forward of Multi Input ConvNet where input_networks
        added via add_input_network"""
        out = multi_input_cnn_add_input_network([
                            torch.ones([10, 1, 28, 28, 28])
                        ])
        assert out.shape == (10, 10)

    def test_forward_pass_not_nan(self, conv3D_net):
        """Confirm out is non nan."""
        test_input = torch.ones([1, *conv3D_net.in_dim])
        test_dataloader = DataLoader(TensorDataset(test_input, test_input))
        output = conv3D_net.forward_pass(
            data_loader=test_dataloader,
            convert_to_class=False)
        assert np.any(~np.isnan(output))

    def test_forward_pass_not_nan(self, conv3D_net_class):
        """Confirm out is non nan."""
        test_input = torch.ones([1, *conv3D_net_class.in_dim])
        test_dataloader = DataLoader(TensorDataset(test_input, test_input))
        raw_output = conv3D_net_class.forward_pass(
            data_loader=test_dataloader,
            convert_to_class=False)
        class_output = conv3D_net_class.metrics.extract_class_labels(
            in_matrix=raw_output)
        assert np.any(~np.isnan(class_output))
        assert np.any(~np.isnan(raw_output))

    def test_freeze_class(self, conv3D_net_class):
        """Test class network freezing."""
        conv3D_net_class.freeze(apply_inputs=False)
        for params in conv3D_net_class.network.parameters():
            assert params.requires_grad is False

    def test_unfreeze_class(self, conv3D_net_class):
        """Test class network unfreezing."""
        conv3D_net_class.freeze(apply_inputs=False)
        conv3D_net_class.unfreeze(apply_inputs=False)
        for params in conv3D_net_class.network.parameters():
            assert params.requires_grad is True

    def test_freeze_noclass(self, conv3D_net):
        """Test intermediate network freezing."""
        conv3D_net.freeze(apply_inputs=False)
        for params in conv3D_net.network.parameters():
            assert params.requires_grad is False

    def test_unfreeze_noclass(self, conv3D_net):
        """Test intermediate network unfreezing."""
        conv3D_net.freeze(apply_inputs=False)
        conv3D_net.unfreeze(apply_inputs=False)
        for params in conv3D_net.network.parameters():
            assert params.requires_grad is True

