"""Test device switching for networks."""
import pytest

import torch
from torch.utils.data import DataLoader, Subset

from vulcanai.models import ConvNet, DenseNet
from vulcanai.models.utils import master_device_setter

TEST_CUDA = torch.cuda.is_available()
TEST_MULTIGPU = TEST_CUDA and torch.cuda.device_count() >= 2
DEVICE_COUNT = 0

if TEST_CUDA:
    DEVICE_COUNT = torch.cuda.device_count()


class TestDevice:
    """Test multi-input GPU device switching."""

    @pytest.mark.skipif(not TEST_CUDA, reason="No CUDA"
                        " supported devices available")
    def test_master_net_device_set_to_cuda(self, multi_input_cnn):
        """Test if the network as whole gets switched to cuda."""
        assert hasattr(multi_input_cnn, 'device')
        master_device_setter(multi_input_cnn, 'cuda:0')
        assert multi_input_cnn.device == torch.device(type='cuda', index=0)
        assert multi_input_cnn.input_networks['conv3D_net']\
            .device == torch.device(type='cuda', index=0)
        assert multi_input_cnn.input_networks['multi_input_dnn']\
            .device == torch.device(type='cuda', index=0)
        assert multi_input_cnn.input_networks['multi_input_dnn'].\
            input_networks['conv1D_net'].\
            device == torch.device(type='cuda', index=0)
        assert multi_input_cnn.input_networks['multi_input_dnn'].\
            input_networks['conv2D_net'].\
            device == torch.device(type='cuda', index=0)

    @pytest.mark.skipif(not TEST_CUDA, reason="No CUDA"
                        " supported devices available")
    def test_fail_mixed_devices(self, multi_input_cnn, conv3D_net,
                                multi_input_dnn, conv1D_net,
                                multi_input_dnn_data,
                                multi_input_cnn_data):
        """Test training throws ValueError when network has mixed devices."""
        assert hasattr(conv1D_net, 'device')
        assert hasattr(conv3D_net, 'device')
        assert hasattr(multi_input_dnn, 'device')
        assert hasattr(multi_input_cnn, 'device')

        master_device_setter(multi_input_cnn, 'cuda:0')
        assert conv3D_net == multi_input_cnn.input_networks['conv3D_net']
        assert multi_input_dnn == multi_input_cnn.input_networks['multi_input_dnn']

        data_len = len(multi_input_cnn_data)
        train_loader = DataLoader(
            Subset(multi_input_cnn_data, range(data_len//2)))
        valid_loader = DataLoader(
            Subset(multi_input_cnn_data, range(data_len//2, data_len)))

        multi_input_cnn.fit(
            train_loader=train_loader,
            val_loader=valid_loader,
            epochs=1,
            plot=False)

        with pytest.raises(ValueError) as e_info:
            multi_input_cnn.input_networks['conv3D_net'].device = 'cpu'
            multi_input_cnn.fit(
                train_loader=train_loader,
                val_loader=valid_loader,
                epochs=1,
                plot=False)

        assert str(e_info.value).endswith("{'conv3D_net': device(type='cpu')}")
