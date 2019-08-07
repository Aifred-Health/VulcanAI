"""Test all ConvNet capabilities."""
import pytest
import numpy as np
import copy
import logging
import os
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

import vulcanai
from vulcanai.models import BaseNetwork
from vulcanai.models.cnn import ConvNet, ConvNetConfig
from vulcanai.models.utils import master_device_setter

logger = logging.getLogger(__name__)


class TestConvNet:
    """Define ConvNet test class."""

    @pytest.fixture
    def multi_input_cnn_train_loader(self, multi_input_cnn_data):
        """Synthetic test data pytorch dataloader object."""
        test_train = Subset(multi_input_cnn_data,
                            range(len(multi_input_cnn_data)//2))
        return DataLoader(test_train, batch_size=2)

    @pytest.fixture
    def multi_input_cnn_test_loader(self, multi_input_cnn_data):
        """Synthetic train data pytorch dataloader object."""
        test_val = Subset(multi_input_cnn_data,
                          range(len(multi_input_cnn_data)//2,
                                len(multi_input_cnn_data)))
        return DataLoader(test_val, batch_size=2)

    def test_init(self, conv1D_net):
        """Initialization Test of a ConvNet object."""
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

    def test_function_multi_input(self, conv1D_net, multi_input_cnn):
        """Test functions wrt multi_input_cnn."""
        tensor_size = [10, 1, 28, 28]
        assert isinstance(multi_input_cnn.input_networks, nn.ModuleDict)
        assert len(list(multi_input_cnn.input_networks)) == 3
        assert all(multi_input_cnn._get_max_incoming_spatial_dims() ==
                   (8, 8, 8))
        # TODO: make more elegant
        assert multi_input_cnn._merge_input_network_outputs([
                torch.rand(size=tensor_size),
                torch.rand(size=tensor_size + [28]),
                torch.rand(size=[10] +
                           list(multi_input_cnn.input_networks
                           ['multi_input_dnn'].out_dim))
        ]).shape == (10, 3, 8, 8, 8)
        test_net = copy.deepcopy(multi_input_cnn)
        test_net._add_input_network(conv1D_net)
        assert len(list(test_net.input_networks)) == 4
        assert 'conv1D_net' in test_net.input_networks
        assert isinstance(test_net.input_networks['conv1D_net'], ConvNet)

    def test_forward(self, conv1D_net):
        """Test Forward of ConvNet."""
        out = conv1D_net(torch.rand(size=[10, *conv1D_net.in_dim]))
        assert out.shape == (10, 64, 1)

    def test_forward_multi_input(self, multi_input_cnn):
        """Test Forward of Multi Input ConvNet."""
        master_device_setter(multi_input_cnn, 'cpu')
        input_tensor = [
            torch.rand(size=[10, 1, 28, 28]),
            torch.rand(size=[10, 1, 28, 28, 28]),
            [torch.rand(size=[10, 1, 28]),
             torch.rand(size=[10, 1, 28, 28])]
        ]
        out = multi_input_cnn(input_tensor)
        assert out.shape == (10, 10)

    def test_forward_pass_not_nan(self, conv3D_net):
        """Confirm out is non nan."""
        test_input = torch.rand(size=[1, *conv3D_net.in_dim])
        test_dataloader = DataLoader(TensorDataset(test_input, test_input))
        output = conv3D_net.forward_pass(
            data_loader=test_dataloader,
            transform_outputs=False)
        assert np.any(~np.isnan(output))

    def test_forward_pass_class_not_nan(self, conv3D_net_class):
        """Confirm out is non nan."""
        test_input = torch.rand(size=[1, *conv3D_net_class.in_dim])
        test_dataloader = DataLoader(TensorDataset(test_input, test_input))
        raw_output = conv3D_net_class.forward_pass(
            data_loader=test_dataloader,
            transform_outputs=False)
        class_output = conv3D_net_class.metrics.transform_outputs(
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

    def test_fit_multi_input(self, multi_input_cnn,
                             multi_input_cnn_train_loader,
                             multi_input_cnn_test_loader):
        """Test for fit function."""
        init_weights = copy.deepcopy(multi_input_cnn.network[0].
                                     _kernel.weight.detach())
        multi_input_cnn_no_fit = copy.deepcopy(multi_input_cnn)
        parameters1 = multi_input_cnn_no_fit.parameters()
        try:
            multi_input_cnn.fit(
                multi_input_cnn_train_loader,
                multi_input_cnn_test_loader,
                2)
        except RuntimeError:
            logger.error("The network multi_input_cnn failed to train.")
        finally:
            parameters2 = multi_input_cnn.parameters()
            trained_weights = multi_input_cnn.network[0]._kernel.weight.detach()

            # Sanity check if the network parameters are training
            assert not (torch.equal(init_weights.cpu(), trained_weights.cpu()))
            compare_params = [not torch.allclose(param1, param2)
                              for param1, param2 in zip(parameters1,
                                                        parameters2)]
            assert all(compare_params)

    def test_params_multi_input(self, multi_input_cnn,
                                multi_input_cnn_train_loader,
                                multi_input_cnn_test_loader):
        """Test for change in network params/specifications."""
        test_net = copy.deepcopy(multi_input_cnn)
        # Check the parameters are copying properly
        copy_params = [torch.allclose(param1, param2)
                       for param1, param2 in zip(multi_input_cnn.parameters(),
                                                 test_net.parameters())]
        assert all(copy_params)

        # Check the parameters change after copy and fit
        test_net.fit(
            multi_input_cnn_train_loader,
            multi_input_cnn_test_loader,
            2)
        close_params = [not torch.allclose(param1, param2)
                        for param1, param2 in zip(multi_input_cnn.parameters(),
                                                  test_net.parameters())]
        assert all(close_params)

        # Check the network params and optimizer params point to
        # the same memory
        if test_net.optim:
            assert isinstance(test_net.optim, torch.optim.Adam)
            assert isinstance(test_net.criterion, torch.nn.CrossEntropyLoss)
            for param, opt_param in zip(test_net.parameters(),
                                        test_net.optim.param_groups[0]['params']):
                assert param is opt_param

        # Check the params after saving loaading
        test_net.save_model()
        save_path = test_net.save_path
        abs_save_path = os.path.dirname(os.path.abspath(save_path))
        loaded_test_net = BaseNetwork.load_model(load_path=save_path)
        load_params = [torch.allclose(param1, param2)
                       for param1, param2 in zip(test_net.parameters(),
                                                 loaded_test_net.parameters())]
        shutil.rmtree(abs_save_path)
        assert all(load_params)

    def test_forward_pass_class_not_nan_single_value(self,
                                                     conv3D_net_class_single_value):
        """Confirm out is non nan."""
        test_input = torch.rand(size=[10, *conv3D_net_class_single_value.in_dim])
        test_output = torch.rand(size=[10, 1])
        test_dataloader = DataLoader(TensorDataset(test_input, test_output))
        raw_output = conv3D_net_class_single_value.forward_pass(
            data_loader=test_dataloader,
            transform_outputs=False)
        assert np.any(~np.isnan(raw_output))

    def test_early_stopping(self, conv3D_net_class_early_stopping,
                            conv3D_net_class):
        """ Test that their final params are different: aka
        that the early stopping did something.
        Constant seed resetting due to worker init function of dataloader
        https://github.com/pytorch/pytorch/issues/7068 """

        vulcanai.set_global_seed(42)

        ds = DataLoader(TensorDataset(
            torch.rand(size=[10, *conv3D_net_class_early_stopping.in_dim]),
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).long()),
            num_workers=0)

        vulcanai.set_global_seed(42)

        ds2 = copy.deepcopy(ds)

        vulcanai.set_global_seed(42)

        ds3 = copy.deepcopy(ds)

        conv3D_net_class_copy = copy.deepcopy(conv3D_net_class)

        conv3D_net_class_copy.fit(
            train_loader=ds2,
            val_loader=ds2,
            epochs=5)
        vulcanai.set_global_seed(42)

        conv3D_net_class.fit(
            train_loader=ds,
            val_loader=ds,
            epochs=5)

        vulcanai.set_global_seed(42)

        conv3D_net_class_early_stopping.fit(
            train_loader=ds3,
            val_loader=ds3,
            epochs=5)

        stopping_params = list(conv3D_net_class_early_stopping.parameters())[0][0][0].data
        non_stopping_params = list(conv3D_net_class.parameters())[0][0][0][0][0].data
        non_stopping_params_copy = list(conv3D_net_class_copy.parameters())[0][0][0][0][0].data

        assert not torch.eq(stopping_params, non_stopping_params).all()
        assert torch.eq(non_stopping_params, non_stopping_params_copy).all()

