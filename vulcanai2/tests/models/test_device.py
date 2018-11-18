import pytest

from vulcanai2.models import ConvNet, DenseNet
from vulcanai2.models.utils import master_device_setter


import torch

TEST_CUDA = torch.cuda.is_available()
TEST_MULTIGPU = TEST_CUDA and torch.cuda.device_count() >= 2
DEVICE_COUNT = 0

if TEST_CUDA:
    DEVICE_COUNT = torch.cuda.device_count()


class TestDevice:
    @pytest.fixture
    def conv1D_net(self):
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
    @pytest.fixture
    def conv2D_net(self):
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

    @pytest.fixture
    def conv3D_net(self):
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

    @pytest.fixture
    def dense_net(self, conv1D_net, conv2D_net):
        return DenseNet(
            name='dense_net',
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

    @pytest.fixture
    def multi_net(self, conv3D_net, dense_net):  
        return ConvNet(
            name='multi_input_network',
            input_networks=[conv3D_net, dense_net],
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

    @pytest.mark.skipif(TEST_CUDA, reason="No CUDA"
                        " supported devices available")
    def test_master_net_device_set_to_cuda(self, multi_net):
        master_device_setter(multi_net, 'cuda:0')
        assert multi_net.device == torch.device(type='cuda', index=0)
        assert multi_net.input_networks['conv3D_net']\
            .device == torch.device(type='cuda', index=0)
        assert multi_net.input_networks['dense_net']\
            .device == torch.device(type='cuda', index=0)
        assert multi_net.input_networks['dense_net'].input_networks['conv1D_net']\
            .device == torch.device(type='cuda', index=0)
        assert multi_net.input_networks['dense_net'].input_networks['conv2D_net']\
            .device == torch.device(type='cuda', index=0)                                    

    # @pytest.mark.xfail
    # def test_fail_mixed_devices(self, multi_net):
    # TODO: write script with mixed devices so that the 
    # network fails when fitting

    # TODO: more tests yet to come

