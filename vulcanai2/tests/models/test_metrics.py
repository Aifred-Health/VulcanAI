import pytest
import numpy as np
import torch
from vulcanai2.models.metrics import Metrics

class TestMetrics:
    @pytest.fixture
    def metrics(self):
        return Metrics(
            num_class=2
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
    
