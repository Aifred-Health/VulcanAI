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
from torch.utils.data import DataLoader, Subset, TensorDataset

from vulcanai2.models import BaseNetwork
from vulcanai2.models.dnn import DenseNet, DenseNetConfig
from vulcanai2.models.utils import master_device_setter

class TestDenseNet:
    """Define DenseNet test class."""

    @pytest.fixture
    def multi_input_train_loader(self, multi_input_dnn_data):
        test_train = Subset(multi_input_dnn_data, 
                            range(len(multi_input_dnn_data)//2))
        return DataLoader(test_train, batch_size=2) 

    @pytest.fixture
    def multi_input_test_loader(self, multi_input_dnn_data):
        test_val = Subset(multi_input_dnn_data, 
                          range(len(multi_input_dnn_data)//2, 
                                    len(multi_input_dnn_data)))
        return DataLoader(test_val, batch_size=2)  

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
        """Test functions wrt multi_input_dnn"""
        assert isinstance(multi_input_dnn.input_networks, nn.ModuleDict)
        assert len(list(multi_input_dnn.input_networks)) == 2
        assert multi_input_dnn._merge_input_network_outputs([
                                torch.ones([10, 1, 28]),
                                torch.ones([10, 1, 28, 28])
               ]).shape == (10, 812)
    
    def test_forward(self, dnn_class):
        """Test Forward of DenseNet"""
        out = dnn_class(torch.ones([10, *dnn_class.in_dim]))
        assert out.shape == (10, 3)
    
    def test_forward_multi_input(self, multi_input_dnn):
        """Test Forward of Multi Input ConvNet"""
        master_device_setter(multi_input_dnn, 'cuda:0')
        input_tensor = [torch.ones([10, 1, 28]),
                        torch.ones([10, 1, 28, 28])]
        out = multi_input_dnn(input_tensor)
        assert out.shape == (10, 50)

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

    # TODO: Add a private function test for _add_input_network
    # TODO: Failing fit! Fix the test
    def test_fit_multi_input(self, multi_input_dnn,
                             multi_input_train_loader,
                             multi_input_test_loader):
        """Test for fit function"""        
        init_weights = copy.deepcopy(multi_input_dnn.network[0]._kernel.weight.detach())
        multi_input_dnn_no_fit = copy.deepcopy(multi_input_dnn)
        parameters1 = multi_input_dnn_no_fit.parameters()
        try:
            multi_input_dnn.fit(multi_input_train_loader, 
                                multi_input_test_loader, 2)
        except RuntimeError:
            logger.error("The network multi_input_dnn failed to train.")
        finally:
            parameters2 = multi_input_dnn.parameters()
            trained_weights = multi_input_dnn.network[0]._kernel.weight.detach()
            
            # Sanity check if the network parameters are training
            assert (torch.equal(init_weights.cpu(), trained_weights.cpu()) is False)
            compare_params = [not torch.allclose(param1, param2)
                        for param1, param2 in zip(parameters1,
                                                  parameters2)]
            assert all(compare_params)
    
    def test_params_multi_input(self, multi_input_dnn,
                                multi_input_train_loader,
                                multi_input_test_loader):
        """Test for change in network params/specifications"""
        
        test_net1 = copy.deepcopy(multi_input_dnn)
        test_net2 = pickle.loads(pickle.dumps(multi_input_dnn))
        
        # Check the parameters are copying properly
        copy_params1 = [torch.allclose(param1, param2)
                        for param1, param2 in zip(multi_input_dnn.parameters(),
                                                  test_net1.parameters())]
        copy_params2 = [torch.allclose(param1, param2)
                        for param1, param2 in zip(multi_input_dnn.parameters(),
                                                  test_net2.parameters())]
        assert all(copy_params1)
        assert all(copy_params2)

        # Check the parameters change after copy and fit
        test_net1.fit(multi_input_train_loader, 
                      multi_input_test_loader, 2)
        test_net2.fit(multi_input_train_loader, 
                      multi_input_test_loader, 2)
        close_params1 = [not torch.allclose(param1, param2)
                        for param1, param2 in zip(multi_input_dnn.parameters(),
                                                  test_net1.parameters())]
        close_params2 = [not torch.allclose(param1, param2)
                        for param1, param2 in zip(multi_input_dnn.parameters(),
                                                  test_net2.parameters())]
        assert all(close_params1)
        assert all(close_params2)

        # Check the network params and optimizer params point to
        # the same memory
        if test_net1.optim:
            assert isinstance(test_net1.optim, torch.optim.Adam)
            assert isinstance(test_net1.criterion, torch.nn.CrossEntropyLoss)
            for param, opt_param in zip(test_net1.parameters(),
                                        test_net1.optim.param_groups[0]['params']):
                assert param is opt_param
        
        # Check the params after saving loaading
        test_net2.save_model()
        save_path = test_net2.save_path
        abs_save_path = os.path.dirname(os.path.abspath(save_path))
        loaded_test_net2 = BaseNetwork.load_model(load_path=save_path)
        load_params = [torch.allclose(param1, param2)
                        for param1, param2 in zip(test_net2.parameters(),
                                                  loaded_test_net2.parameters())]
        shutil.rmtree(abs_save_path)
        assert all(load_params)
