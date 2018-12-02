import pytest

from vulcanai2.models import ConvNet, DenseNet

import torch


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
        }
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
def multi_input_cnn(conv3D_net, multi_input_dnn):
    """Bottom multi-input network fixture."""
    return ConvNet(
        name='multi_input_cnn',
        input_networks=[conv3D_net, multi_input_dnn],
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
def cnn_noclass():
    """Create ConvNet with no prediction layer."""
    return ConvNet(
        name='Test_ConvNet_noclass',
        in_dim=(1, 28, 28),
        config={
            'conv_units': [
                {
                    "in_channels": 1,
                    "out_channels": 16,
                    "kernel_size": (5, 5),
                    "pool_size": 2,
                    "stride": 2
                },
                {
                    "in_channels": 16,
                    "out_channels": 1,
                    "kernel_size": (5, 5),
                    "stride": 2,
                    "padding": 2
                }]
        }
    )

@pytest.fixture(scope="module")
def cnn_class():
    """Create ConvNet with prediction layer."""
    return ConvNet(
        name='Test_ConvNet_class',
        in_dim=(1, 28, 28),
        config={
            'conv_units': [
                {
                    "in_channels": 1,
                    "out_channels": 16,
                    "kernel_size": (5, 5),
                    "pool_size": 2,
                    "stride": 2
                },
                {
                    "in_channels": 16,
                    "out_channels": 1,
                    "kernel_size": (5, 5),
                    "stride": 2,
                    "padding": 2
                }]
        },
        num_classes=3
    )

@pytest.fixture(scope="module")
def cnn_class_add_input_network(cnn_noclass, cnn_class):
    """Create ConvNet with input_network added via
    add_input_network and has a prediction layer."""
    net = cnn_class
    net.add_input_network(cnn_noclass)
    return net