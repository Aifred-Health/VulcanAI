import pytest
import numpy as np

import torch
import torch.nn as nn
from torch.nn import modules

from vulcanai2.models.layers import *

class TestBaseUnit:
    @pytest.fixture
    def baseunit(self): 
        return BaseUnit()

    def test_init(self, baseunit):
        """Initialization Test of a BaseUnit object"""
        assert isinstance(baseunit, nn.Sequential)
        assert hasattr(baseunit, 'initializer')
        assert hasattr(baseunit, 'bias_init')
        assert hasattr(baseunit, 'norm')
        assert hasattr(baseunit, 'dropout')
        assert hasattr(baseunit, 'in_shape')
        assert hasattr(baseunit, 'out_shape')
        assert hasattr(baseunit, '_kernel')

class TestDenseUnit:
    @pytest.fixture
    def denseunit(self):
        return DenseUnit(
            in_features=10,
            out_features=10,
            initializer=nn.init.xavier_uniform_,
            bias_init=0.1,
            norm='batch',
            activation=nn.ReLU(),
            dropout=0.5
        )
    
    def test_init(self, denseunit):
        """Initialization Test of a DenseUnit object"""
        assert hasattr(denseunit, 'in_features')
        assert hasattr(denseunit, 'out_features')
        for unit in denseunit.named_children():
            assert any(unit[0]==i for i in ['_kernel', '_norm', 
                       '_activation', '_dropout'])
            assert isinstance(unit[1], (nn.Linear, 
                              nn.modules.batchnorm._BatchNorm, 
                              nn.ReLU, modules.dropout._DropoutNd))
            assert callable(unit[1])
    
    def test_forward(self, denseunit):
        """Confirm size is expected after forward."""
        test_input = torch.ones([10, denseunit.in_features])
        output = denseunit.forward(test_input)
        assert output.shape == torch.ones([10, denseunit.out_features]).size()
    

class TestConvUnit:
    @pytest.fixture
    def convunit(self):
        return ConvUnit(
            conv_dim=2,
            in_channels=10,
            out_channels=10,
            kernel_size=(5,5),
            initializer=nn.init.xavier_uniform_,
            bias_init=0.1,
            norm='batch',
            activation=nn.ReLU(),
            dropout=0.5
        )

    def test_init(self, convunit):
        """Initialization Test of a ConvUnit object"""
        assert hasattr(convunit, 'in_channels')
        assert hasattr(convunit, 'out_channels')
        assert hasattr(convunit, 'kernel_size')
        for unit in convunit.named_children():
            assert any(unit[0]==i for i in ['_kernel', '_norm', 
                       '_activation', '_dropout'])
            assert isinstance(unit[1], (modules.conv._ConvNd, 
                              nn.modules.batchnorm._BatchNorm, 
                              nn.ReLU, modules.dropout._DropoutNd))
            assert callable(unit[1])
    
    def test_forward(self, convunit):
        """Confirm size is expected after forward."""
        test_input = torch.ones([10, convunit.in_channels, 28, 28])
        output = convunit.forward(test_input)
        # No padding with 2 5x5 kernels leads from 28x28 -> 24x24
        assert output.shape == torch.ones([10, convunit.out_channels, 24, 24]).size()

