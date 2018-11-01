import pytest
import numpy as np
import torch
from vulcanai2.models.layers import *

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

