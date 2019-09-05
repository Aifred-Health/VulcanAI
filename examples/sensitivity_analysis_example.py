from torch.utils.data import DataLoader, Subset, TensorDataset
from vulcanai.models import ConvNet, DenseNet
import sys
sys.path.append('.')
from vulcanai.models.metrics import Metrics
import torch

def dataloader():
    """Create a dataset by importing from the test csv"""
    test_input = torch.rand(size=[13, 15])
    test_dataloader = DataLoader(
        TensorDataset(test_input, torch.tensor([0, 0, 0, 1, 2, 0, 2, 0, 0, 0, 0,
                                                0, 0])))
    return test_dataloader

def dnn():
    """DenseNet with prediction layer."""
    return DenseNet(
        name='dnn_class',
        in_dim=(15),
        config={
            'dense_units': [100, 50],
            'dropout': 0.5,
        },
        num_classes=4
    )

if __name__ == '__main__':
    net = dnn()
    dl = dataloader()

    m = Metrics()

    # with col names given
    m.conduct_sensitivity_analysis(net, dl, 'test_sensitivity_analysis_1',
                                   ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                                    'i', 'j', 'k', 'l'])

    # without col names given
    m.conduct_sensitivity_analysis(net, dl, 'test_sensitivity_analysis_2')
