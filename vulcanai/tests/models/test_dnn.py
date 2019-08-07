"""Test all DenseNet capabilities."""
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
from vulcanai.models.dnn import DenseNet, DenseNetConfig
from vulcanai.models.utils import master_device_setter

logger = logging.getLogger(__name__)


class TestDenseNet:
    """Define DenseNet test class."""

    @pytest.fixture
    def multi_input_dnn_train_loader(self, multi_input_dnn_data):
        """Synthetic train data pytorch dataloader object."""
        test_train = Subset(multi_input_dnn_data,
                            range(len(multi_input_dnn_data)//2))
        return DataLoader(test_train, batch_size=2)

    @pytest.fixture
    def multi_input_dnn_test_loader(self, multi_input_dnn_data):
        """Synthetic test data pytorch dataloader object."""
        test_val = Subset(multi_input_dnn_data,
                          range(len(multi_input_dnn_data)//2,
                                len(multi_input_dnn_data)))
        return DataLoader(test_val, batch_size=2)

    def test_init(self, dnn_noclass):
        """Initialization Test of a DenseNet object."""
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
        assert dnn_noclass.optim is None
        assert dnn_noclass.criterion is None

        assert not hasattr(dnn_noclass, 'metrics')

    def test_function_multi_input(self, dnn_noclass,
                                  multi_input_dnn):
        """Test functions wrt multi_input_dnn."""
        assert isinstance(multi_input_dnn.input_networks, nn.ModuleDict)
        assert len(list(multi_input_dnn.input_networks)) == 2
        assert multi_input_dnn._merge_input_network_outputs([
                torch.rand(size=[10, 1, 28]),
                torch.rand(size=[10, 1, 28, 28])
            ]).shape == (10, 812)
        test_net = copy.deepcopy(multi_input_dnn)
        test_net._add_input_network(dnn_noclass)
        assert len(list(test_net.input_networks)) == 3
        assert 'dnn_noclass' in test_net.input_networks
        assert isinstance(test_net.input_networks['dnn_noclass'], DenseNet)

    def test_forward(self, dnn_class):
        """Test Forward of DenseNet."""
        out = dnn_class(torch.rand(size=[10, *dnn_class.in_dim]))
        assert out.shape == (10, 3)

    def test_forward_multi_input(self, multi_input_dnn):
        """Test Forward of Multi Input ConvNet."""
        master_device_setter(multi_input_dnn, 'cpu')
        input_tensor = [torch.rand(size=[10, 1, 28]),
                        torch.rand(size=[10, 1, 28, 28])]
        out = multi_input_dnn(input_tensor)
        assert out.shape == (10, 50)

    def test_forward_pass_not_nan(self, dnn_noclass):
        """Confirm out is non nan."""
        test_input = torch.rand(size=[5, *dnn_noclass.in_dim])
        test_dataloader = DataLoader(TensorDataset(test_input, test_input))
        output = dnn_noclass.forward_pass(
            data_loader=test_dataloader,
            transform_outputs=False)
        assert np.any(~np.isnan(output))

    def test_forward_pass_class_not_nan(self, dnn_class):
        """Confirm out is non nan."""
        test_input = torch.rand(size=[5, *dnn_class.in_dim])
        test_dataloader = DataLoader(TensorDataset(test_input, test_input))
        raw_output = dnn_class.forward_pass(
            data_loader=test_dataloader,
            transform_outputs=False)
        class_output = dnn_class.metrics.transform_outputs(
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

    def test_fit_multi_input(self, multi_input_dnn_class,
                             multi_input_dnn_train_loader,
                             multi_input_dnn_test_loader):
        """Test for fit function."""
        init_weights = copy.deepcopy(multi_input_dnn_class.network[0].
                                     _kernel.weight.detach())
        multi_input_dnn_class_no_fit = copy.deepcopy(multi_input_dnn_class)
        parameters1 = multi_input_dnn_class_no_fit.parameters()
        try:
            multi_input_dnn_class.fit(
                multi_input_dnn_train_loader,
                multi_input_dnn_test_loader,
                2)
        except RuntimeError:
            logger.error("The network multi_input_dnn_class failed to train.")
        finally:
            parameters2 = multi_input_dnn_class.parameters()
            trained_weights = multi_input_dnn_class.network[0]._kernel.weight.detach()

            # Sanity check if the network parameters are training
            # We want to be sure weights are different.
            # Hacked so that we can be sure we're properly comparing floats
            # There is no negation of np.testing.assert_almost_equal.
            # Thus we throw an error if an error doesn't occur from checking eq
            weights_same = True
            try:
                np.testing.assert_almost_equal(init_weights.cpu().numpy(),
                                               trained_weights.cpu().numpy())
            except AssertionError:
                weights_same = False

            if weights_same:
                raise AssertionError

            #assert not (torch.equal(init_weights.cpu(), trained_weights.cpu()))
            compare_params = [not torch.allclose(param1, param2)
                              for param1, param2 in zip(parameters1,
                                                        parameters2)]
            assert all(compare_params)

    def test_params_multi_input(self, multi_input_dnn_class,
                                multi_input_dnn_train_loader,
                                multi_input_dnn_test_loader):
        """Test for change in network params/specifications."""
        test_net = copy.deepcopy(multi_input_dnn_class)
        # Check the parameters are copying properly
        copy_params = [torch.allclose(param1, param2)
                       for param1, param2 in zip(multi_input_dnn_class.parameters(),
                                                 test_net.parameters())]
        assert all(copy_params)

        # Check the parameters change after copy and fit
        test_net.fit(
            multi_input_dnn_train_loader,
            multi_input_dnn_test_loader,
            2)
        close_params = [not torch.allclose(param1, param2)
                        for param1, param2 in zip(multi_input_dnn_class.parameters(),
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
                                                     dnn_class_single_value):
        """Confirm out is non nan."""
        test_input = torch.rand(size=[10, *dnn_class_single_value.in_dim])
        test_output = torch.rand(size=[10, 1])
        test_dataloader = DataLoader(TensorDataset(test_input, test_output))
        raw_output = dnn_class_single_value.forward_pass(
            data_loader=test_dataloader,
            transform_outputs=False)
        assert np.any(~np.isnan(raw_output))

    def test_early_stopping(self, dnn_class_early_stopping,
                            dnn_class):
        """ Test that their final params are different: aka
        that the early stopping did something
        Constant seed resetting due to worker init function of dataloader
        https://github.com/pytorch/pytorch/issues/7068 """

        ds = DataLoader(TensorDataset(
            torch.rand(size=[3, *dnn_class_early_stopping.in_dim]),
            torch.tensor([0, 1, 2]).long()))

        vulcanai.set_global_seed(42)

        ds2 = copy.deepcopy(ds)

        vulcanai.set_global_seed(42)

        ds3 = copy.deepcopy(ds)

        dnn_class_copy = copy.deepcopy(dnn_class)

        dnn_class_copy.fit(
            train_loader=ds2,
            val_loader=ds2,
            epochs=5)

        vulcanai.set_global_seed(42)

        dnn_class.fit(
            train_loader=ds,
            val_loader=ds,
            epochs=5)

        vulcanai.set_global_seed(42)

        dnn_class_early_stopping.fit(
            train_loader=ds3,
            val_loader=ds3,
            epochs=5)

        stopping_params = list(dnn_class_early_stopping.parameters())[0][0].data
        non_stopping_params = list(dnn_class.parameters())[0][0].data
        non_stopping_params_copy = list(dnn_class_copy.parameters())[0][0].data

        assert not torch.eq(stopping_params, non_stopping_params).all()
        assert torch.eq(non_stopping_params, non_stopping_params_copy).all()

