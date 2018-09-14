import pytest
import numpy as np
import torch
from vulcanai2.models.layers import *

class TestBaseLayer:
    @pytest.fixture
    def baselayer(self): 
        return BaseLayer()

    # TODO: Needs fixing assigning values to NoneType
    # def test_update_batch_size(self, baselayer):
    #     """Confirm out is non nan."""
    #     baselayer.update_batch_size(31)
    #     assert baselayer.batch_size == 31
    #     assert baselayer.in_shape[0] == 31
    #     assert baselayer.out_shape[0] == 31

class TestDenseUnit:
    @pytest.fixture
    def dense_unit(self):
        return DenseUnit(
            in_channels=10,
            out_channels=10
        )
    
    def test_forward(self, dense_unit):
        """Confirm size is expected after forward."""
        test_input = torch.ones([1, dense_unit.in_channels])
        output = dense_unit.forward(test_input)
        assert output.size() == torch.ones([1, dense_unit.out_channels]).size()
    

class TestConvUnit:
    @pytest.fixture
    def conv_unit(self):
        return ConvUnit(
            conv_dim=2,
            in_channels=10,
            out_channels=10
        )
    
    def test_forward(self, conv_unit):
        """Confirm size is expected after forward."""
        test_input = torch.ones([1, conv_unit.in_channels, 28, 28])
        output = conv_unit.forward(test_input)
        # Get's padded by 2 in every dim by default
        assert output.size() == torch.ones([1, conv_unit.out_channels, 30, 30]).size()

