import pytest


class TestBaseNetwork:
    @pytest.fixture
    def basenet(self):
        from vulcanai2.models.basenetwork import BaseNetwork
        return BaseNetwork(
            name='Test_BaseNet',
            dimensions=(None, 10),
            config={}
        )

    def test_name(self, basenet):
        basenet.name = 'New_Name'
        assert basenet.name == 'New_Name'
    
    def test_learning_rate(self, basenet):
        basenet.learning_rate = 0.1
        assert basenet.learning_rate == 0.1
