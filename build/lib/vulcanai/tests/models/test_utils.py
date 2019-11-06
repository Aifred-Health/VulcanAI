"""Define the model utilitiy tests."""
import pytest
import numpy as np
import torch
from vulcanai.models.utils import (round_list,
                                    get_one_hot,
                                    pad,
                                    set_tensor_device,
                                    master_device_setter)

TEST_CUDA = torch.cuda.is_available()


def test_round_list():
    """Test if the list is rounded to the desired decimals."""
    test_list = [0.83754245, 0.13249807]
    out = round_list(test_list, decimals=3)
    assert len(str(out[0]).split(".")[1]) == 3


def test_get_one_hot():
    """Test for get_one_hot."""
    test_inp = np.array([0, 1, 2, 3, 4, 3, 2, 1, 0])
    assert np.all(get_one_hot(test_inp) == np.array([
                    [1., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0.],
                    [0., 0., 1., 0., 0.],
                    [0., 0., 0., 1., 0.],
                    [0., 0., 0., 0., 1.],
                    [0., 0., 0., 1., 0.],
                    [0., 0., 1., 0., 0.],
                    [0., 1., 0., 0., 0.],
                    [1., 0., 0., 0., 0.]
                ]))


def test_pad():
    """Test if input_tensor is padded to the desired padded shape."""
    n_channels = 2
    n_features = 50
    pad_shape = 250
    batch = 1
    test_inp = torch.randn([batch, n_channels, n_features])
    padded_tensor = pad(tensor=test_inp, target_shape=[pad_shape])
    assert padded_tensor.shape == (batch, n_channels, pad_shape)
    assert padded_tensor.nonzero().size(0) == n_channels * n_features
    # 2D padding
    pad_shape1 = 100
    test_inp = torch.randn([batch, n_channels, n_features, n_features])
    padded_tensor = pad(
        tensor=test_inp,
        target_shape=[pad_shape, pad_shape1])
    assert padded_tensor.shape == (batch, n_channels, pad_shape, pad_shape1)
    assert padded_tensor.nonzero().size(0) == \
        n_channels * n_features * n_features
    # 3D padding
    pad_shape2 = 50
    test_inp = torch.randn([batch, n_channels, n_features, n_features,
                            n_features])
    padded_tensor = pad(
        tensor=test_inp,
        target_shape=[pad_shape, pad_shape1, pad_shape2])
    assert padded_tensor.shape == \
        (batch, n_channels, pad_shape, pad_shape1, pad_shape2)
    assert padded_tensor.nonzero().size(0) == \
        n_channels * n_features * n_features * n_features


@pytest.mark.skipif(not TEST_CUDA, reason="No CUDA"
                    " supported devices available")
def test_set_tensor_device():
    """If CUDA is available, test set_tensor_device usage."""
    # Tensor
    test_tensor_cpu = torch.randn([5, 5]).cpu()
    test_tensor_cuda = set_tensor_device(test_tensor_cpu, device='cuda:0')
    assert str(test_tensor_cuda.device) == 'cuda:0'

    # List of Tensors
    test_tensorlist_cpu = [torch.randn([1, 2]).cpu(),
                           torch.randn([2, 3]).cpu(),
                           torch.randn([3, 4]).cpu()]
    test_tensorlist_cuda = set_tensor_device(
        test_tensorlist_cpu, device='cuda:0')
    for t in test_tensorlist_cuda:
        assert str(t.device) == 'cuda:0'


@pytest.mark.skipif(not TEST_CUDA, reason="No CUDA"
                    " supported devices available")
def test_master_device_setter(multi_input_cnn):
    """If CUDA is available, test master_device_setter usage."""
    # Make sure the network is in cpu first
    assert str(multi_input_cnn.device) == 'cpu'
    master_device_setter(multi_input_cnn, device='cuda:0')
    assert str(multi_input_cnn.device) == 'cuda:0'
    assert str(list(multi_input_cnn.input_networks.values())[0] == 'cuda:0')
    assert str(list(list(multi_input_cnn.input_networks.
                    values())[2].input_networks.values())[0] == 'cuda:0')
