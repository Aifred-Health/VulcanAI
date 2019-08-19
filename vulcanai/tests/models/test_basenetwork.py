"""Test BaseNetwork functionality."""
import pytest
from vulcanai.models.basenetwork import BaseNetwork

import torch


class TestBaseNetwork:
    """Define BaseNetwork test class."""

    @pytest.fixture
    def basenet(self):
        """Create a test BaseNetwork."""
        return BaseNetwork(
            name='Test_BaseNet',
            in_dim=(None, 10),
            config={}
        )

    def test_init(self, basenet):
        """Initialization Test of a BaseNetwork object."""
        assert isinstance(basenet, BaseNetwork)
        assert isinstance(basenet, torch.nn.Module)
        assert hasattr(basenet, 'network')
        assert hasattr(basenet, 'in_dim')
        assert hasattr(basenet, 'record')

    def test_name(self, basenet):
        """Test changing names."""
        basenet.name = 'New_Name'
        assert basenet.name is 'New_Name'

    def test_learning_rate(self, basenet):
        """Test learning rate change."""
        basenet.learning_rate = 0.1
        assert basenet.learning_rate is 0.1

    def test_default_criter_spec(self, basenet):
        """Test default value behaviour for criter spec"""
        assert isinstance(basenet.criter_spec, torch.nn.CrossEntropyLoss)
        assert isinstance(basenet._final_transform, torch.nn.Softmax)
