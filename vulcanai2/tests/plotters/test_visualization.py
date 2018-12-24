"""Will test visualization functions."""
import pytest
import numpy as np
import torch
from copy import deepcopy
import os
from torch.utils.data import TensorDataset, DataLoader
from vulcanai2.plotters.visualization import compute_saliency_map, \
                                             display_receptive_fields


class TestVisualization:
    """Test all visualization functionality."""

    @pytest.fixture
    def cnn_class(self):
        """Create ConvNet with classes fixture."""
        from vulcanai2.models.cnn import ConvNet
        return ConvNet(
            name='Test_ConvNet_class',
            in_dim=(1, 28, 28),
            config={
                'conv_units': [
                    {
                        "in_channels": 1,
                        "out_channels": 16,
                        "kernel_size": (5, 5),
                        "stride": 1,
                        "padding": 2
                    },
                    {
                        "in_channels": 16,
                        "out_channels": 1,
                        "kernel_size": (5, 5),
                        "stride": 1,
                        "padding": 2
                    }]
            },
            num_classes=3
        )

    @pytest.fixture
    def dnn_class(self):
        """Create DenseNet with classes fixture."""
        from vulcanai2.models.dnn import DenseNet
        return DenseNet(
            name='Test_DenseNet_class',
            in_dim=(200),
            config={
                'dense_units': [100, 50],
                'dropouts': 0.3,
            },
            num_classes=3
        )

    @pytest.fixture
    def dnn_class_two(self):
        """Create DenseNet with no prediction layer."""
        from vulcanai2.models.dnn import DenseNet
        return DenseNet(
            name='Test_DenseNet_class',
            in_dim=(200),
            activation=torch.nn.SELU(),
            num_classes=2,
            input_networks=None,
            config={
                'dense_units': [100],
                'dropout': [0.3],
                'initializer': None,
                'bias_init': None,
                'norm': None
            },
            optim_spec={'name': 'Adam', 'lr': 0.001}
        )

    def test_compute_saliency_map_cnn(self, cnn_class):
        """Confirm hooks are removed, and gradient shape."""
        test_input_1B = torch.ones([1, *cnn_class.in_dim])
        test_input_5B = torch.ones([5, *cnn_class.in_dim])

        model_copy = deepcopy(cnn_class)
        # Test shape conservation
        cnn_class.freeze(apply_inputs=False)
        sal_map_1B = compute_saliency_map(
            cnn_class,
            test_input_1B, torch.LongTensor([2]))
        for sal_map, test_input in zip(sal_map_1B, [test_input_1B]):
            assert sal_map.shape == test_input.shape

        cnn_class.unfreeze(apply_inputs=False)
        sal_map_5B = compute_saliency_map(
            cnn_class, test_input_5B,
            torch.LongTensor([0, 2, 1, 1, 0]))
        for sal_map, test_input in zip(sal_map_5B, [test_input_5B]):
            assert sal_map.shape == test_input.shape

        # Check that all gradients are not 0
        assert ~np.all(sal_map_5B == 0.)

        # Test hook removal
        assert cnn_class._backward_hooks == model_copy._backward_hooks

    def test_compute_saliency_map_dnn(self, dnn_class):
        """Confirm hooks are removed, and gradient shape."""
        test_input_1B = torch.ones([1, *dnn_class.in_dim])
        test_input_5B = torch.ones([5, *dnn_class.in_dim])

        model_copy = deepcopy(dnn_class)
        # Test shape conservation
        dnn_class.freeze(apply_inputs=False)
        sal_map_1B = compute_saliency_map(
            dnn_class,
            test_input_1B, torch.LongTensor([2]))
        for sal_map, test_input in zip(sal_map_1B, [test_input_1B]):
            assert sal_map.shape == test_input.shape

        dnn_class.unfreeze(apply_inputs=False)
        sal_map_5B = compute_saliency_map(
            dnn_class, test_input_5B,
            torch.LongTensor([0, 2, 1, 1, 0]))
        for sal_map, test_input in zip(sal_map_5B, [test_input_5B]):
            assert sal_map.shape == test_input.shape

        # Check that all gradients are not 0
        assert ~np.all(sal_map_5B == 0.)

        # Test hook removal
        assert dnn_class._backward_hooks == model_copy._backward_hooks

    def test_receptive_field(self, dnn_class_two):
        curr_path = str(os.path.dirname(__file__)) + '/'
        test_input = torch.ones([10, *dnn_class_two.in_dim]).float()
        test_target = torch.LongTensor([0, 1, 1, 0, 0, 1, 0, 1, 0 , 1])
        test_dataloader = DataLoader(TensorDataset(test_input, test_target))

        dnn_class_two.fit(test_dataloader, test_dataloader, 5)
        display_receptive_fields(dnn_class_two)
        for file in os.listdir(curr_path):
            if file.startswith('feature_importance'):
                assert True
                file_path = curr_path + file
                os.remove(file_path)