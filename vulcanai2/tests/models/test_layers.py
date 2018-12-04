import pytest
import numpy as np
import math
import copy
import torch
from vulcanai2.models.layers import *
from vulcanai2.models.utils import selu_weight_init_, selu_bias_init_

class TestBaseUnit:
    @pytest.fixture
    def baseunit(self): 
        return BaseUnit()

class TestDenseUnit:
    @pytest.fixture
    def dense_unit(self):
        return DenseUnit(
            in_features=10,
            out_features=10
        )
    
    def test_forward(self, dense_unit):
        """Confirm size is expected after forward."""
        test_input = torch.ones([1, dense_unit.in_features])
        output = dense_unit.forward(test_input)
        assert output.size() == torch.ones([1, dense_unit.out_features]).size()
    

class TestConvUnit:
    @pytest.fixture
    def conv_unit(self):
        return ConvUnit(
            conv_dim=2,
            in_channels=10,
            out_channels=10,
            kernel_size=(5,5)
        )
    
    def test_forward(self, conv_unit):
        """Confirm size is expected after forward."""
        test_input = torch.ones([1, conv_unit.in_channels, 28, 28])
        output = conv_unit.forward(test_input)
        # No padding with 2 5x5 kernels leads from 28x28 -> 24x24
        assert output.size() == torch.ones([1, conv_unit.out_channels, 24, 24]).size()

class TestSeluWeightInit:
    @pytest.fixture
    def dense_unit(self):
        return DenseUnit(in_features=10,
                         out_features=10
        )

    @pytest.fixture
    def conv_unit(self):
        return ConvUnit(
            conv_dim=2,
            in_channels=10,
            out_channels=10,
            kernel_size=(5,5)
        )

    def test_dense_selu_weight_change(self, dense_unit):
        starting_weight = copy.deepcopy(dense_unit._kernel.weight)
        std = math.sqrt(1. / 10)
        #Produce upper and lower by finding values that are 3 standard deviations from mean
        # Statistically, ~99% of values in normal distribution are 3 std away from mean.
        upper = 3 * std
        lower = 3 * (-(math.sqrt(1. / 10)))


        dense_unit.weight_init = selu_weight_init_
        dense_unit.weight_init(dense_unit._kernel.weight)
        new_weight = copy.deepcopy(dense_unit._kernel.weight)
        assert ((torch.equal(starting_weight, new_weight) is False) and
               (new_weight.max().item() <= upper) and
               (new_weight.min().item() >= lower))

    def test_conv_selu_weight_change(self, conv_unit):
        starting_weight = copy.deepcopy(conv_unit._kernel.weight)
        std = math.sqrt(1. / 10)
        #Produce upper and lower by finding values that are 3 standard deviations from mean
        # Statistically, ~99% of values in normal distribution are 3 std away from mean.
        upper = 3 * std
        lower = 3 * (-(math.sqrt(1. / 10)))

        conv_unit.weight_init = selu_weight_init_
        conv_unit.weight_init(conv_unit._kernel.weight)
        new_weight = copy.deepcopy(conv_unit._kernel.weight)
        assert ((torch.equal(starting_weight, new_weight) is False) and
               (new_weight.max().item() <= upper) and
               (new_weight.min().item() >= lower))

    def test_dense_selu_bias_change(self, dense_unit):
        starting_bias = copy.deepcopy(dense_unit._kernel.bias)
        std = math.sqrt(1. / 10)
        #Produce upper and lower by finding values that are 3 standard deviations from mean
        # Statistically, ~99% of values in normal distribution are 3 std away from mean.
        upper = 3 * std
        lower = 3 * (-(math.sqrt(1. / 10)))


        dense_unit.bias_init = selu_bias_init_
        dense_unit.bias_init(dense_unit._kernel.bias)
        new_bias = copy.deepcopy(dense_unit._kernel.bias)
        assert ((torch.equal(starting_bias, new_bias) is False) and
               (new_bias.max().item() <= upper) and
               (new_bias.min().item() >= lower))

    def test_conv_selu_bias_change(self, conv_unit):
        starting_bias = copy.deepcopy(conv_unit._kernel.bias)
        std = math.sqrt(1. / 10)
        #Produce upper and lower by finding values that are 3 standard deviations from mean
        # Statistically, ~99% of values in normal distribution are 3 std away from mean.
        upper = 3 * std
        lower = 3 * (-(math.sqrt(1. / 10)))


        conv_unit.bias_init = selu_bias_init_
        conv_unit.bias_init(conv_unit._kernel.bias)
        new_bias = copy.deepcopy(conv_unit._kernel.bias)
        assert ((torch.equal(starting_bias, new_bias) is False) and
               (new_bias.max().item() <= upper) and
               (new_bias.min().item() >= lower))