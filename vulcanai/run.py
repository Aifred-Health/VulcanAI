from torch.utils.data import DataLoader, Subset
from vulcanai.datasets import TabularDataset
from vulcanai.models import ConvNet, DenseNet
import sys
sys.path.append('../')
from models.metrics import conduct_sensitivity_analysis

def my_test_dataset():
    """Create a dataset by importing from the test csv"""
    fpath = "/Users/snehadesai/Documents/Vulcan/vulcanai/tests/datasets/test_data/birthweight_reduced.csv"
    return TabularDataset(
        data=fpath,
        label_column="id",
        na_values='Nan'
    )

def tabular_dataset_train_loader(my_test_dataset):
    """Test csv data pytorch dataloader object."""
    test_train = Subset(my_test_dataset,
                        range(len(my_test_dataset)//2))
    return DataLoader(test_train, batch_size=2)

def dnn_class_single_value_2():
    """DenseNet with prediction layer."""
    return DenseNet(
        name='dnn_class',
        in_dim=(12),
        config={
            'dense_units': [100, 50],
            'dropout': 0.5,
        },
        num_classes=22
    )



def run_sensitivity_analysis(dnn_class_single_value_2,
                              tabular_dataset_train_loader):
    conduct_sensitivity_analysis(dnn_class_single_value_2,
                                         tabular_dataset_train_loader,
                                         'test_sensitivity_analysis')


if __name__ == '__main__':
    net = dnn_class_single_value_2()
    dl = tabular_dataset_train_loader(my_test_dataset())
    run_sensitivity_analysis(net, dl)
