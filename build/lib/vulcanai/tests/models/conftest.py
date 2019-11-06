"""Specify dummy networks to test vulcan functionality."""
import pytest

import torch
from torch.utils.data import TensorDataset
from vulcanai.datasets import MultiDataset
from vulcanai.models import ConvNet, DenseNet
from torch.utils.data import DataLoader, Subset


@pytest.fixture(scope="module")
def conv1D_net():
    """conv1D fixture."""
    return ConvNet(
        name='conv1D_net',
        in_dim=(1, 28),
        config={
            'conv_units': [
                dict(
                    in_channels=1,
                    out_channels=24,
                    kernel_size=(5),
                    stride=2,
                    pool_size=2,
                    dropout=0.1
                ),
                dict(
                    in_channels=24,
                    out_channels=64,
                    kernel_size=(5),
                    pool_size=2,
                    dropout=0.1
                )
            ],
        },
        device='cpu'
    )


@pytest.fixture(scope="module")
def conv2D_net():
    """conv2D fixture."""
    return ConvNet(
        name='conv2D_net',
        in_dim=(1, 28, 28),
        config={
            'conv_units': [
                dict(
                    in_channels=1,
                    out_channels=24,
                    kernel_size=(5, 5),
                    stride=2,
                    pool_size=2,
                    dropout=0.1
                ),
                dict(
                    in_channels=24,
                    out_channels=64,
                    kernel_size=(5, 5),
                    pool_size=2,
                    dropout=0.1
                )
            ],
        },
        device='cpu'
    )


@pytest.fixture(scope="module")
def conv3D_net():
    """conv3D fixture."""
    return ConvNet(
        name='conv3D_net',
        in_dim=(1, 28, 28, 28),
        config={
            'conv_units': [
                dict(
                    in_channels=1,
                    out_channels=16,
                    kernel_size=(5, 5, 5),
                    stride=2,
                    dropout=0.1
                ),
                dict(
                    in_channels=16,
                    out_channels=64,
                    kernel_size=(5, 5, 5),
                    dropout=0.1
                )
            ],
        },
        device='cpu'
    )


@pytest.fixture(scope="module")
def conv3D_net_class():
    """conv3D fixture."""
    return ConvNet(
        name='conv3D_net',
        in_dim=(1, 28, 28, 28),
        num_classes=10,
        config={
            'conv_units': [
                dict(
                    in_channels=1,
                    out_channels=16,
                    kernel_size=(5, 5, 5),
                    stride=2,
                    dropout=0.1
                ),
                dict(
                    in_channels=16,
                    out_channels=64,
                    kernel_size=(5, 5, 5),
                    dropout=0.1
                )
            ],
        },
        device='cpu'
    )

@pytest.fixture(scope="module")
def conv3D_net_class_early_stopping():
    """conv3D fixture."""
    return ConvNet(
        name='conv3D_net',
        in_dim=(1, 28, 28, 28),
        num_classes=10,
        early_stopping="best_validation_error",
        early_stopping_patience=2,
        config={
            'conv_units': [
                dict(
                    in_channels=1,
                    out_channels=16,
                    kernel_size=(5, 5, 5),
                    stride=2,
                    dropout=0.1
                ),
                dict(
                    in_channels=16,
                    out_channels=64,
                    kernel_size=(5, 5, 5),
                    dropout=0.1
                )
            ],
        },
        device='cpu'
    )

@pytest.fixture(scope="module")
def conv3D_net_class_single_value():
    """conv3D fixture."""
    return ConvNet(
        name='conv3D_net',
        in_dim=(1, 28, 28, 28),
        num_classes=1,
        config={
            'conv_units': [
                dict(
                    in_channels=1,
                    out_channels=16,
                    kernel_size=(5, 5, 5),
                    stride=2,
                    dropout=0.1
                ),
                dict(
                    in_channels=16,
                    out_channels=64,
                    kernel_size=(5, 5, 5),
                    dropout=0.1
                )
            ],
        },
        device='cpu'
    )


@pytest.fixture(scope="module")
def dnn_noclass():
    """DenseNet fixture."""
    return DenseNet(
        name='dnn_noclass',
        in_dim=(200),
        config={
            'dense_units': [100, 50],
            'dropout': [0.3, 0.5],
        }
    )


@pytest.fixture(scope="module")
def dnn_class():
    """DenseNet with prediction layer."""
    return DenseNet(
        name='dnn_class',
        in_dim=(200),
        config={
            'dense_units': [100, 50],
            'dropout': 0.5,
        },
        num_classes=3
    )


@pytest.fixture(scope="module")
def dnn_class_early_stopping():
    """DenseNet with prediction layer."""
    return DenseNet(
        name='dnn_class',
        in_dim=(200),
        early_stopping="best_validation_error",
        early_stopping_patience=2,
        config={
            'dense_units': [100, 50],
            'dropout': 0.5,
        },
        num_classes=3
    )


@pytest.fixture(scope="module")
def dnn_class_single_value():
    """DenseNet with prediction layer."""
    return DenseNet(
        name='dnn_class',
        in_dim=(200),
        config={
            'dense_units': [100, 50],
            'dropout': 0.5,
        },
        num_classes=1
    )

@pytest.fixture(scope="module")
def multi_input_dnn(conv1D_net, conv2D_net):
    """Dense network fixture with two inputs."""
    return DenseNet(
        name='multi_input_dnn',
        input_networks=[conv1D_net, conv2D_net],
        config={
            'dense_units': [100, 50],
            'initializer': None,
            'bias_init': None,
            'norm': None,
            'dropout': 0.5,  # Single value or List
        },
        device='cpu'
    )


@pytest.fixture(scope="module")
def multi_input_cnn(conv2D_net, conv3D_net, multi_input_dnn):
    """Bottom multi-input network fixture."""
    return ConvNet(
        name='multi_input_cnn',
        input_networks=[conv2D_net, conv3D_net, multi_input_dnn],
        num_classes=10,
        config={
            'conv_units': [
                dict(
                    in_channels=1,
                    out_channels=16,
                    kernel_size=(3, 3, 3),
                    stride=2,
                    dropout=0.1
                ),
            ],
        },
        device='cpu'
    )


@pytest.fixture(scope="module")
def multi_input_dnn_class(conv1D_net, conv2D_net):
    """Dense network fixture with two inputs."""
    return DenseNet(
        name='multi_input_dnn_class',
        input_networks=[conv1D_net, conv2D_net],
        num_classes=10,
        config={
            'dense_units': [100, 50],
            'initializer': None,
            'bias_init': None,
            'norm': None,
            'dropout': 0.5,  # Single value or List
        },
        device='cpu'
    )


@pytest.fixture(scope="module")
def multi_input_dnn_data(conv1D_net, conv2D_net,
                         multi_input_dnn):
    return MultiDataset([
        (
            TensorDataset(
                torch.rand(size=[10, *conv1D_net.in_dim]),
                torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).long()),
            True, True),
        (
            TensorDataset(
                torch.rand(
                    size=[
                        10,
                        *multi_input_dnn.input_networks['conv2D_net'].
                        in_dim])),
            True, False)
    ])


@pytest.fixture(scope="module")
def multi_input_cnn_data(conv2D_net, conv3D_net, multi_input_dnn_data):
    return MultiDataset([
        (TensorDataset(torch.rand(size=[10, *conv2D_net.in_dim])), True, False),
        (TensorDataset(torch.rand(size=[10, *conv3D_net.in_dim])), True, False),
        multi_input_dnn_data
    ])


@pytest.fixture(scope="module")
def dnn_class_multi_value():
    """DenseNet with prediction layer and mulitple classes."""
    return DenseNet(
        name='dnn_class',
        in_dim=(12),
        config={
            'dense_units': [100, 50],
            'dropout': 0.5,
        },
        num_classes=3
    )


@pytest.fixture(scope="module")
def sensitivity_data_loader():
    torch.manual_seed(7)
    test_input = torch.rand(size=[5, 12])
    test_dataloader = DataLoader(TensorDataset(test_input, torch.tensor(
        [0, 1, 2, 0, 1])))
    return test_dataloader
