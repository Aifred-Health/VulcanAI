import pytest
import numpy as np
import torch
from vulcanai2.models.dnn import DenseNet
from torch.utils.data import TensorDataset, DataLoader

class TestDenseNet:
    @pytest.fixture
    def dnn_noclass(self):
        return DenseNet(
            name='Test_DenseNet_class',
            in_dim=(200),
            config={
                'dense_units': [100, 50],
                'dropout': [0.3, 0.5],
            }
        )
    
    @pytest.fixture
    def dnn_class(self):
        return DenseNet(
            name='Test_DenseNet_class',
            in_dim=(200),
            config={
                'dense_units': [100, 50],
                'dropout': 0.5,
            },
            num_classes=3
        )

    def test_forward_not_nan(self, dnn_noclass):
        """Confirm out is non nan."""
        test_input = torch.ones([5, *dnn_noclass.in_dim])
        test_dataloader = DataLoader(TensorDataset(test_input, test_input))
        output = dnn_noclass.forward_pass(
            data_loader=test_dataloader,
            convert_to_class=False)
        assert np.any(~np.isnan(output))
    
    def test_forward_class_not_nan(self, dnn_class):
        """Confirm out is non nan."""
        test_input = torch.ones([5, *dnn_class.in_dim])
        test_dataloader = DataLoader(TensorDataset(test_input, test_input))
        raw_output = dnn_class.forward_pass(
            data_loader=test_dataloader,
            convert_to_class=False)
        class_output = dnn_class.metrics.get_class(
            in_matrix=test_input)
        assert np.any(~np.isnan(raw_output))
        assert np.any(~np.isnan(class_output))

