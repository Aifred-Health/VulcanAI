"""Test BaseNetwork functionality."""
import pytest
from vulcanai2.models.basenetwork import BaseNetwork


class TestBaseNetwork:
    """Define BaseNetwork test class."""

    @pytest.fixture
    def basenet(self):
        """Create a test BaseNetwork."""
        return BaseNetwork(
            name='Test_BaseNet',
            dimensions=(None, 10),
            config={}
        )

    def test_name(self, basenet):
        """Test changing names."""
        basenet.name = 'New_Name'
        assert basenet.name is 'New_Name'

    def test_learning_rate(self, basenet):
        """Test learning rate change."""
        basenet.learning_rate = 0.1
        assert basenet.learning_rate is 0.1
