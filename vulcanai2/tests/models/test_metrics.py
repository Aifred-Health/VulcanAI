import pytest
import numpy as np
import torch
from vulcanai2.models.metrics import Metrics

class TestDenseUnit:
    @pytest.fixture
    def metrics(self):
        return Metrics(
            num_class=2
        )
    
    def test_get_class(self, metrics):
        """Correctly represents max likelihood class."""
        test_input = torch.Tensor(
            [
                [0.2, 0.8],
                [0.7, 0.3],
                [0.25, 0.75]
            ]
        )
        output = metrics.get_class(test_input)
        assert np.all(output == [[1], [0], [1]])
    
