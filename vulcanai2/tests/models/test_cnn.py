import pytest
import numpy as np
import torch
from vulcanai2.models.cnn import ConvNet
from torch.utils.data import TensorDataset, DataLoader

class TestConvNet:
    @pytest.fixture
    def cnn_noclass(self): 
        return ConvNet(
            name='Test_ConvNet_noclass',
            in_dim=(1, 28, 28),
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
            in_dim=(1, 28, 28),
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
        test_input = torch.ones([1, *cnn_noclass.in_dim[0]])
        test_dataloader = DataLoader(TensorDataset(test_input, test_input))
        output = cnn_noclass.forward_pass(
            data_loader=test_dataloader,
            convert_to_class=False)
        assert np.any(~np.isnan(output))
    
    def test_forward_class_not_nan(self, cnn_class):
        """Confirm out is non nan."""
        test_input = torch.ones([1, *cnn_class.in_dim[0]])
        test_dataloader = DataLoader(TensorDataset(test_input, test_input))
        raw_output = cnn_class.forward_pass(
            data_loader=test_dataloader,
            convert_to_class=False)
        class_output = cnn_class.metrics.get_class(
            in_matrix=raw_output)
        assert np.any(~np.isnan(class_output))
        assert np.any(~np.isnan(raw_output))

