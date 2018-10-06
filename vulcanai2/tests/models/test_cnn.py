import pytest
import numpy as np
import torch
from vulcanai2.models.cnn import ConvNet

class TestConvNet:
    @pytest.fixture
    def cnn_noclass(self): 
        return ConvNet(
            name='Test_ConvNet_noclass',
            dimensions=(1, 28, 28),
            config={
                'conv_units': [
                    {
                        "in_channels":1,
                        "out_channels":16,
                        "kernel_size":(5, 5),
                        "stride":2
                    },
                    {
                        "in_channels":16,
                        "out_channels":1,
                        "kernel_size":(5, 5),
                        "stride":1,
                        "padding":2
                    }]
            }
        )

    @pytest.fixture
    def cnn_class(self):
        return ConvNet(
            name='Test_ConvNet_class',
            dimensions=(1, 28, 28),
            config={
                'conv_units': [
                    {
                        "in_channels":1,
                        "out_channels":16,
                        "kernel_size":(5, 5),
                        "stride":2
                    },
                    {
                        "in_channels":16,
                        "out_channels":1,
                        "kernel_size":(5, 5),
                        "stride":1,
                        "padding":2
                    }]
            },
            num_classes=3
        )

    def test_forward_not_nan(self, cnn_noclass):
        """Confirm out is non nan."""
        test_input = np.ones([1, *cnn_noclass.in_dim])
        output = cnn_noclass.forward_pass(test_input)
        assert np.any(~np.isnan(output))
    
    def test_forward_class_not_nan(self, cnn_class):
        """Confirm out is non nan."""
        test_input = np.ones([1, *cnn_class.in_dim])
        output = cnn_class.forward_pass(test_input)
        assert np.any(~np.isnan(output))

