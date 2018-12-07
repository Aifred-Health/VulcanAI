"""Test all ConvNet capabilities."""
import numpy as np
import pytest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from vulcanai2.models import BaseNetwork
from vulcanai2.models.cnn import ConvNet
from vulcanai2.models.utils import master_device_setter


class TestConvNet:
    """Define ConvNet test class."""
    
    def test_init(self, conv1D_net):
        """Initialization Test of ConvNet """
        assert isinstance(conv1D_net, BaseNetwork)
        assert isinstance(conv1D_net, nn.Module)
        assert hasattr(conv1D_net, 'input_networks')
        assert hasattr(conv1D_net, 'network')
        assert hasattr(conv1D_net, 'in_dim')
        assert hasattr(conv1D_net, 'device')
    
    def test_forward(self, conv1D_net):
        """Test Forward of ConvNet"""
        out = conv1D_net(torch.ones([10, *conv1D_net.in_dim]))
        assert out.shape == (10, 64, 1)

    def test_forward_multi_input(self, multi_input_cnn,
                                         multi_input_cnn_data):
        """Test Forward of Multi Input ConvNet"""
        master_device_setter(multi_input_cnn, 'cuda:0')
        out = multi_input_cnn([torch.ones([10, 1, 28, 28]),
                               torch.ones([10, 1, 28, 28, 28]),
                               [torch.ones([10, 1, 28]),
                                torch.ones([10, 1, 28, 28])]
                                ])
        assert out.shape == (10, 10)
    
    def test_forward_multi_input_cnn_add_input_network(self, 
                                multi_input_cnn_add_input_network):
        """Test Forward of Multi Input ConvNet where input_networks
        added via add_input_network"""
        out = multi_input_cnn_add_input_network([
                            torch.ones([10, 1, 28]),
                            torch.ones([10, 1, 28, 28]),
                            torch.ones([10, 1, 28, 28, 28])
                        ])
        assert out.shape == (10, 10)

    def test_forward_pass_not_nan(self, cnn_noclass):
        """Confirm out is non nan."""
        test_input = torch.ones([1, *cnn_noclass.in_dim])
        test_dataloader = DataLoader(TensorDataset(test_input, test_input))
        output = cnn_noclass.forward_pass(
            data_loader=test_dataloader,
            convert_to_class=False)
        assert np.any(~np.isnan(output))

    def test_forward_pass_not_nan(self, cnn_class):
        """Confirm out is non nan."""
        test_input = torch.ones([1, *cnn_class.in_dim])
        test_dataloader = DataLoader(TensorDataset(test_input, test_input))
        raw_output = cnn_class.forward_pass(
            data_loader=test_dataloader,
            convert_to_class=False)
        class_output = cnn_class.metrics.extract_class_labels(
            in_matrix=raw_output)
        assert np.any(~np.isnan(class_output))
        assert np.any(~np.isnan(raw_output))

    def test_freeze_class(self, cnn_class):
        """Test class network freezing."""
        cnn_class.freeze(apply_inputs=False)
        for params in cnn_class.network.parameters():
            assert params.requires_grad is False

    def test_unfreeze_class(self, cnn_class):
        """Test class network unfreezing."""
        cnn_class.freeze(apply_inputs=False)
        cnn_class.unfreeze(apply_inputs=False)
        for params in cnn_class.network.parameters():
            assert params.requires_grad is True

    def test_freeze_noclass(self, cnn_noclass):
        """Test intermediate network freezing."""
        cnn_noclass.freeze(apply_inputs=False)
        for params in cnn_noclass.network.parameters():
            assert params.requires_grad is False

    def test_unfreeze_noclass(self, cnn_noclass):
        """Test intermediate network unfreezing."""
        cnn_noclass.freeze(apply_inputs=False)
        cnn_noclass.unfreeze(apply_inputs=False)
        for params in cnn_noclass.network.parameters():
            assert params.requires_grad is True
            
    # TODO: temporarily cutting this test off
    # def test_add_input_network(self, cnn_class_add_input_network,
    #                            cnn_noclass):
    #     """Test add input Network functionality."""
    #     assert isinstance(cnn_class_add_input_network.input_networks,
    #                       nn.ModuleDict)
    #     assert cnn_class_add_input_network\
    #            .input_networks[cnn_noclass.name] is cnn_noclass
    #     assert cnn_class_add_input_network.in_dim == cnn_noclass.out_dim
