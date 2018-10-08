import pytest
import numpy as np
import torch
from copy import deepcopy

from vulcanai2.plotters.visualization import compute_saliency_map

class TestVisualization:

    @pytest.fixture
    def cnn_class(self):
        from vulcanai2.models.cnn import ConvNet
        return ConvNet(
            name='Test_ConvNet_class',
            dimensions=(1, 28, 28),
            config={
                'conv_units': [
                    {
                        "in_channels":1,
                        "out_channels":16,
                        "kernel_size":(5, 5),
                        "stride":1,
                        "padding":2
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

    @pytest.fixture
    def dnn_class(self):
        from vulcanai2.models.dnn import DenseNet
        return DenseNet(
            name='Test_DenseNet_class',
            dimensions=(200),
            config={
                'dense_units': [100, 50],
                'dropouts': 0.3,
            },
            num_classes=3
        )

    def test_compute_saliency_map_cnn(self, cnn_class):
        """
        Confirm hooks are removed, and gradient shape is the same as input.
        For ConvNets.
        """
        test_input_1B = np.ones([1, *cnn_class.in_dim], dtype=np.float32)
        test_input_5B = np.ones([5, *cnn_class.in_dim], dtype=np.float32)

        model_copy = deepcopy(cnn_class)
        # Test shape conservation
        sal_map_1B = compute_saliency_map(
            cnn_class,
            test_input_1B, torch.tensor([2]))
        assert sal_map_1B.shape == test_input_1B.shape
        sal_map_5B = compute_saliency_map(
            cnn_class, test_input_5B,
            torch.tensor([0, 2, 1, 1, 0]))
        assert sal_map_5B.shape == test_input_5B.shape

        # Check that all gradients are not 0
        assert ~np.all(sal_map_5B == 0.)

        # Test hook removal
        assert cnn_class._backward_hooks == model_copy._backward_hooks

    def test_compute_saliency_map_dnn(self, dnn_class):
        """
        Confirm hooks are removed, and gradient shape is the same as input.
        For DenseNets.
        """
        test_input_1B = np.ones([1, dnn_class.in_dim], dtype=np.float32)
        test_input_5B = np.ones([5, dnn_class.in_dim], dtype=np.float32)

        model_copy = deepcopy(dnn_class)
        # Test shape conservation
        sal_map_1B = compute_saliency_map(
            dnn_class,
            test_input_1B, torch.tensor([2]))
        assert sal_map_1B.shape == test_input_1B.shape
        sal_map_5B = compute_saliency_map(
            dnn_class, test_input_5B,
            torch.tensor([0, 2, 1, 1, 0]))
        assert sal_map_5B.shape == test_input_5B.shape

        # Check that all gradients are not 0
        assert ~np.all(sal_map_5B == 0.)

        # Test hook removal
        assert dnn_class._backward_hooks == model_copy._backward_hooks
