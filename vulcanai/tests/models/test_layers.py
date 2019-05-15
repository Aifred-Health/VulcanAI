"""The script to test all layers and SELU activation properties hold."""
import pytest
import numpy as np
import math
import copy
from functools import reduce
import torch
import torch.nn as nn
from torch.nn import modules
from vulcanai.models.layers import BaseUnit, ConvUnit, DenseUnit
from vulcanai.models.utils import selu_weight_init_, selu_bias_init_
from vulcanai.models.dnn import DenseNet
from vulcanai.models.cnn import ConvNet
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(1234)

class TestBaseUnit:
    """To test BaseUnit layer."""

    @pytest.fixture
    def baseunit(self):
        """Make Base unit fixture."""
        return BaseUnit()

    def test_init(self, baseunit):
        """Initialization Test of a BaseUnit object."""
        assert isinstance(baseunit, nn.Sequential)
        assert hasattr(baseunit, 'weight_init')
        assert hasattr(baseunit, 'bias_init')
        assert hasattr(baseunit, 'norm')
        assert hasattr(baseunit, 'dropout')
        assert hasattr(baseunit, 'in_shape')
        assert hasattr(baseunit, 'out_shape')
        assert hasattr(baseunit, '_kernel')


class TestDenseUnit:
    """To test DenseUnit layer."""

    @pytest.fixture
    def denseunit(self):
        """Dense unit fixture."""
        return DenseUnit(
            in_features=10,
            out_features=10,
            weight_init=nn.init.xavier_uniform_,
            bias_init=nn.init.zeros_,
            norm='batch',
            activation=nn.ReLU(),
            dropout=0.5
        )

    def test_init(self, denseunit):
        """Initialization Test of a DenseUnit object."""
        assert hasattr(denseunit, 'in_features')
        assert hasattr(denseunit, 'out_features')
        for unit in denseunit.named_children():
            assert any(unit[0] == i for i in ['_kernel', '_norm',
                       '_activation', '_dropout'])
            assert isinstance(unit[1], (nn.Linear,
                              nn.modules.batchnorm._BatchNorm,
                              nn.ReLU, modules.dropout._DropoutNd))
            assert callable(unit[1])

    @pytest.fixture
    def test_denseunit_parameters(self):
        """Create a dictionary with incorrect DenseUnit parameters."""
        return dict(
            in_channels=10,
            out_features=10,
            weight_init=nn.init.xavier_uniform_,
            bias_init=nn.init.zeros_,
            norm='batch',
            activation=nn.ReLU(),
            dropout=0.5
        )

    def test_create_denseunit(self, test_denseunit_parameters):
        """Check if passing wrong parameters raises TypeError."""
        with pytest.raises(TypeError) as e:
            DenseUnit(**test_denseunit_parameters)
        assert 'in_channels' in str(e.value)

    def test_forward(self, denseunit):
        """Confirm size is expected after forward."""
        test_input = torch.rand(size=[10, denseunit.in_features])
        output = denseunit.forward(test_input)
        assert output.shape == torch.rand(size=[10, denseunit.out_features]).shape


class TestConvUnit:
    """To test ConvUnit layers."""

    @pytest.fixture
    def convunit(self):
        """Create ConvUnit fixture."""
        return ConvUnit(
            conv_dim=2,
            in_channels=10,
            out_channels=10,
            kernel_size=(5, 5),
            weight_init=nn.init.xavier_uniform_,
            bias_init=nn.init.zeros_,
            norm='batch',
            activation=nn.ReLU(),
            dropout=0.5
        )

    @pytest.fixture
    def test_convunit_parameters(self):
        """Create dictionary with incorrect ConvUnit parameters."""
        return dict(
            conv_dim=2,
            in_channels=10,
            out_features=10,
            kernel_size=(5, 5),
            weight_init=nn.init.xavier_uniform_,
            bias_init=nn.init.zeros_,
            norm='batch',
            activation=nn.ReLU(),
            dropout=0.5
        )

    def test_create_convunit(self, test_convunit_parameters):
        """Check if passing wrong parameters raises TypeError."""
        with pytest.raises(TypeError) as e:
            ConvUnit(**test_convunit_parameters)
        assert 'out_features' in str(e.value)

    def test_init(self, convunit):
        """Initialization Test of a ConvUnit object."""
        assert hasattr(convunit, 'in_channels')
        assert hasattr(convunit, 'out_channels')
        assert hasattr(convunit, 'kernel_size')
        for unit in convunit.named_children():
            assert any(unit[0] == i for i in
                       ['_kernel', '_norm', '_activation', '_dropout'])
            assert isinstance(unit[1], (modules.conv._ConvNd,
                              nn.modules.batchnorm._BatchNorm,
                              nn.ReLU, modules.dropout._DropoutNd))
            assert callable(unit[1])

    def test_forward(self, convunit):
        """Confirm size is expected after forward."""
        test_input = torch.rand(size=[10, convunit.in_channels, 28, 28])
        output = convunit.forward(test_input)
        # No padding with 2 5x5 kernels leads from 28x28 -> 24x24
        assert output.shape == \
               torch.rand(size=[10, convunit.out_channels, 24, 24]).shape


class TestSeluInit:
    """To test selu initialized layer properties."""

    @pytest.fixture
    def dense_unit(self):
        """Create a dense unit fixture."""
        return DenseUnit(
            in_features=10,
            out_features=10
        )

    @pytest.fixture
    def conv_unit(self):
        """Create a conv unit fixture."""
        return ConvUnit(
            conv_dim=2,
            in_channels=10,
            out_channels=10,
            kernel_size=(5, 5)
        )

    def test_dense_selu_weight_change(self, dense_unit):
        """Confirm SELU weight init properties hold for dense net."""
        starting_weight = copy.deepcopy(dense_unit._kernel.weight)
        fan_in = dense_unit.in_features
        std = round(math.sqrt(1. / fan_in), 1)

        dense_unit.weight_init = selu_weight_init_
        dense_unit._init_weights()
        new_weight = dense_unit._kernel.weight
        assert (torch.equal(starting_weight, new_weight) is False)
        assert (math.isclose(new_weight.std().item(), math.sqrt(1. / fan_in), rel_tol=std) is True)
        assert (int(new_weight.mean().item()) == 0.0)

    def test_conv_selu_weight_change(self, conv_unit):
        """Confirm SELU weight init properties hold for conv net."""
        starting_weight = copy.deepcopy(conv_unit._kernel.weight)
        fan_in = conv_unit._kernel.in_channels * \
            reduce(lambda k1, k2: k1 * k2, conv_unit._kernel.kernel_size)
        std = math.sqrt(1. / fan_in)
        conv_unit.weight_init = selu_weight_init_
        conv_unit._init_weights()
        new_weight = conv_unit._kernel.weight
        assert (torch.equal(starting_weight, new_weight) is False)
        assert (math.isclose(new_weight.std().item(), math.sqrt(1./fan_in), rel_tol=std) is True)
        assert (int(new_weight.mean().item()) == 0)

    def test_dense_selu_bias_change(self, dense_unit):
        """Confirm SELU bias init properties hold for dense net."""
        starting_bias = copy.deepcopy(dense_unit._kernel.bias)

        dense_unit.bias_init = selu_bias_init_
        dense_unit._init_bias()
        new_bias = dense_unit._kernel.bias
        assert (torch.equal(starting_bias, new_bias) is False)
        assert (round(new_bias.std().item(), 1) == 0.0)
        assert (int(new_bias.mean().item()) == 0)

    def test_conv_selu_bias_change(self, conv_unit):
        """Confirm SELU bias init properties hold for conv net."""
        starting_bias = copy.deepcopy(conv_unit._kernel.bias)

        conv_unit.bias_init = selu_bias_init_
        conv_unit._init_bias()
        new_bias = conv_unit._kernel.bias
        assert (torch.equal(starting_bias, new_bias) is False)
        assert (round(new_bias.std().item(), 1) == 0.0)
        assert (int(new_bias.mean().item()) == 0)


class TestSeluInitTrain:
    """To test selu initialization properties hold during training."""

    @pytest.fixture
    def dnn_class(self):
        """Create DenseNet with no prediction layer."""
        return DenseNet(
            name='Test_DenseNet_class',
            in_dim=(200),
            activation=torch.nn.SELU(),
            num_classes=10,
            config={
                'dense_units': [100],
                'dropout': [0.3],
            },
            optim_spec={'name': 'Adam', 'lr': 0.001}
        )

    @pytest.fixture
    def cnn_class(self):
        """Create ConvNet with prediction layer."""
        return ConvNet(
            name='Test_ConvNet_class',
            in_dim=(1, 28, 28),
            activation=torch.nn.SELU(),
            config={
                'conv_units': [
                    {
                        "in_channels": 1,
                        "out_channels": 16,
                        "kernel_size": (5, 5),
                        "stride": 2
                    },
                    {
                        "in_channels": 16,
                        "out_channels": 1,
                        "kernel_size": (5, 5),
                        "stride": 1,
                        "padding": 2
                    }]
            },
            num_classes=10
        )

    def test_selu_trained_dense(self, dnn_class):
        """Confirm SELU weight and bias properties hold for a dense net."""
        fan_in = dnn_class.in_dim[0]
        std = round(math.sqrt(1. / fan_in), 1)
        test_input = torch.rand(size=[10, *dnn_class.in_dim]).float()
        test_target = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        test_dataloader = DataLoader(TensorDataset(test_input, test_target))
        dnn_class.fit(test_dataloader, test_dataloader, 5)
        trained = dnn_class.network.dense_0._kernel
        assert (round(trained.weight.std().item(), 1) == std)
        assert (int(trained.weight.mean().item()) == 0.0)
        assert (round(trained.bias.std().item(), 1) == 0.0)
        assert (int(trained.bias.mean().item()) == 0.0)

    def test_selu_trained_conv(self, cnn_class):
        """Confirm SELU weight and bias properties hold for a conv net."""
        fan_in = cnn_class.network[0].in_channels * \
            reduce(lambda k1, k2: k1 * k2, cnn_class.network[0].kernel_size)
        std = round(math.sqrt(1. / fan_in), 1)
        test_input = torch.rand(size=[10, *cnn_class.in_dim]).float()
        test_target = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        test_dataloader = DataLoader(TensorDataset(test_input, test_target))
        cnn_class.fit(test_dataloader, test_dataloader, 5)
        trained = cnn_class.network.conv_0._kernel
        assert (round(trained.weight.std().item(), 1) == std)
        assert (int(trained.weight.mean().item()) == 0.0)
        assert (round(trained.bias.std().item(), 1) == 0.0)
        assert (int(trained.bias.mean().item()) == 0.0)
