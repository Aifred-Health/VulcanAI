import pytest
import numpy as np
import torch
from vulcanai2.models.metrics import Metrics
from vulcanai2.models.cnn import ConvNet
from vulcanai2.models.dnn import DenseNet
from vulcanai2.models.ensemble import SnapshotNet
from torch.utils.data import TensorDataset, DataLoader

class TestMetrics:

    #note that the number of classes needs to stay consistent

    @pytest.fixture
    def metrics(self):
        return Metrics()

    @pytest.fixture
    def cnn_class(self):
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
                        "stride": 2
                    },
                    {
                        "in_channels": 16,
                        "out_channels": 1,
                        "kernel_size": (5, 5),
                        "stride": 1,
                        "padding": 2
                    }]
            },
            num_classes=6
        )
    
    def test_get_class(self, metrics):
        """Correctly represents max likelihood class."""
        test_input = np.array([
                [0.2, 0.8],
                [0.7, 0.3],
                [0.25, 0.75]
            ])
        output = metrics.get_class(test_input)
        assert np.all(output == np.array([1, 0, 1]))

    def test_cross_validate_outputs(self, metrics, cnn_class):
        """Tests that the cross-validate outputs are in the correct form."""
        test_input = torch.ones([13, *cnn_class.in_dim]).float()
        test_target = torch.LongTensor([0, 2, 1, 3, 4, 1, 2, 2, 3, 0, 4, 5, 0])
        test_dataloader = DataLoader(TensorDataset(test_input, test_target))

        k = 2
        epochs = 2

        averaged_results = metrics.cross_validate(
            cnn_class, test_dataloader, k, epochs,
            average_results=True)
        all_results = metrics.cross_validate(
            cnn_class, test_dataloader, k, epochs,
            average_results=False)

        for k in averaged_results:
            assert isinstance(averaged_results[k], float)

        for k in all_results:
            assert isinstance(all_results[k], list)