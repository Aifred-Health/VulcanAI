"""Will test visualization functions."""
import pytest
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from vulcanai.datasets import tabular_data_utils
import os
from copy import deepcopy

from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from vulcanai.plotters.visualization import compute_saliency_map, \
                                            display_pca, \
                                            display_tsne, \
                                            display_receptive_fields, \
                                            display_saliency_overlay, \
                                            display_confusion_matrix


class TestVisualization:
    """Test all visualization functionality."""

    @pytest.fixture
    def cnn_class(self):
        """Create ConvNet with classes fixture."""
        from vulcanai.models.cnn import ConvNet
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
        from vulcanai.models.dnn import DenseNet
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
        from vulcanai.models.dnn import DenseNet
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

    def test_display_saliency_overlay(self, dnn_class):
        """Test saliency overlay displays and saves."""
        curr_path = str(os.path.dirname(__file__)) + '/'
        img = np.zeros((25,25))
        test_input_1B = torch.ones([1, *dnn_class.in_dim])
        sal_map_1B = np.array(compute_saliency_map(
            dnn_class,
            test_input_1B, torch.LongTensor([2])))
        display_saliency_overlay(img, sal_map_1B, shape=(25,25), save_path=curr_path)
        file_created = False
        for file in os.listdir(curr_path):
            if file.startswith('saliency'):
                file_created = True
                file_path = curr_path + file
                os.remove(file_path)
        assert file_created

    def test_display_pca(self, dnn_class):
        """Test PCA displays and saves."""
        curr_path = str(os.path.dirname(__file__)) + '/'
        digits = load_digits()
        display_pca(
            digits.data[0:10], digits.target[0:10], save_path=curr_path)
        file_created = False
        for file in os.listdir(curr_path):
            if file.startswith('PCA'):
                file_created = True
                file_path = curr_path + file
                os.remove(file_path)
        assert file_created

    def test_display_tsne(self, dnn_class):
        """Test t-SNE displays and saves."""
        curr_path = str(os.path.dirname(__file__)) + '/'
        digits = load_digits()
        display_tsne(
            digits.data[0:10], digits.target[0:10], save_path=curr_path)
        file_created = False
        for file in os.listdir(curr_path):
            if file.startswith('t-SNE'):
                file_created = True
                file_path = curr_path + file
                os.remove(file_path)
        assert file_created

    def test_display_confusion_matrix(self):
        """Test confusion matrix displays and saves."""
        curr_path = str(os.path.dirname(__file__)) + '/'
        cm = confusion_matrix(y_true=[0, 1, 1, 0, 0], y_pred=[1, 1, 0, 1, 0])
        display_confusion_matrix(cm, class_list=[0, 1], save_path=curr_path)
        file_created = False
        for file in os.listdir(curr_path):
            if file.startswith('confusion'):
                file_created = True
                file_path = curr_path + file
                os.remove(file_path)
        assert file_created

    def test_receptive_field(self, dnn_class_two):
        """Test receptive field visualization gets created and can be saved."""
        curr_path = str(os.path.dirname(__file__)) + '/'
        test_input = torch.ones([10, *dnn_class_two.in_dim]).float()
        test_target = torch.LongTensor([0, 1, 1, 0, 0, 1, 0, 1, 0, 1])
        test_dataloader = DataLoader(TensorDataset(test_input, test_target))

        dnn_class_two.fit(test_dataloader, test_dataloader, 5)
        display_receptive_fields(dnn_class_two, save_path=curr_path)
        file_created = False
        for file in os.listdir(curr_path):
            if file.startswith('feature_importance'):
                file_created = True
                file_path = curr_path + file
                os.remove(file_path)
        assert file_created
