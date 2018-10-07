import pytest
import numpy as np
import torch
from vulcanai2.models.dnn import DenseNet

class TestDenseNet:
    @pytest.fixture
    def dnn_noclass(self):
        return DenseNet(
            name='Test_DenseNet_class',
            dimensions=(200),
            config={
                'dense_units': [100],
                'dropout': [0.3],
            }
        )
    
    @pytest.fixture
    def dnn_class(self):
        return DenseNet(
            name='Test_DenseNet_class',
            dimensions=(200),
            config={
                'dense_units': [100],
                'dropout': [0.3],
            },
            num_classes=3
        )

    def test_forward_not_nan(self, dnn_noclass):
        """Confirm out is non nan."""
        test_input = np.ones([1, dnn_noclass.in_dim])
        output = dnn_noclass.forward_pass(test_input)
        assert np.any(~np.isnan(output))
    
    def test_forward_class_not_nan(self, dnn_class):
        """Confirm out is non nan."""
        test_input = np.ones([1, dnn_class.in_dim])
        raw_output = dnn_class.forward_pass(test_input)
        class_output = dnn_class.forward_pass(test_input, convert_to_class=True)
        assert np.any(~np.isnan(raw_output))
        assert np.any(~np.isnan(class_output))

