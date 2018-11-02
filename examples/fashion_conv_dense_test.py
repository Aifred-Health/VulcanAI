import sys
sys.path.append('../')
from vulcanai2 import models, datasets, plotters
from vulcanai2.models import ConvNet, DenseNet, SnapshotNet
from vulcanai2.datasets import MultiDataset
from vulcanai2.plotters.visualization import (compute_saliency_map, 
                                              display_saliency_overlay,
                                              display_receptive_fields,
                                              display_confusion_matrix)
from vulcanai2.models.utils import get_confusion_matrix


import pickle
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
# from torchviz import make_dot

import numpy as np
import json
from collections import OrderedDict as odict
import matplotlib.pyplot as plt

from vulcanai2.models.metrics import Metrics

sys.path.append('../')
normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])


data_path = "../data"
train_dataset = datasets.FashionData(root=data_path,
                            train=True,
                            transform=transform,
                            download=True
                           )

train_dataset = torch.utils.data.Subset(train_dataset, range(0,1000))

val_dataset = datasets.FashionData(root=data_path,
                            train=False,
                            transform=transform,
                           )


batch_size = 100

train_loader = DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

val_loader = DataLoader(dataset=val_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


conv_1D_config = {
    'conv_units': [
                    dict(
                        in_channels=1,
                        out_channels=16,
                        kernel_size=(5),
                        stride=2, # Makes a big difference in training speeds
                        dropout=0.1 # Float or None
                    ),
                    dict(
                        in_channels=16,
                        out_channels=32,
                        kernel_size=(5),
                        padding=0,
                        dropout=0.1 # Float or None
                    ),
                    dict(
                        in_channels=32,
                        out_channels=64,
                        kernel_size=(5),
                        pool_size=2,
                        dropout=0.1 # Float or None
                        )
    ],
}
conv_2D_config = {
    'conv_units': [
                    dict(
                        in_channels=1,
                        out_channels=16,
                        kernel_size=(5, 5),
                        stride=2, # Makes a big difference in training speeds
                        dropout=0.1 # Float or None
                    ),
                    dict(
                        in_channels=16,
                        out_channels=32,
                        kernel_size=(5, 5),
                        dropout=0.1 # Float or None
                    ),
                    dict(
                        in_channels=32,
                        out_channels=64,
                        kernel_size=(5, 5),
                        pool_size=2,
                        dropout=0.1 # Float or None
                        )
    ],
}
conv_3D_config = {
    'conv_units': [
                    dict(
                        in_channels=1,
                        out_channels=16,
                        kernel_size=(5, 5, 5),
                        stride=2, # Makes a big difference in training speeds
                        dropout=0.1 # Float or None
                    ),
                    dict(
                        in_channels=16,
                        out_channels=16,
                        kernel_size=(5, 5, 5),
                        stride=1, # Makes a big difference in training speeds
                        dropout=0.1 # Float or None
                    ),
                    dict(
                        in_channels=16,
                        out_channels=64,
                        kernel_size=(5, 5, 5),
                        dropout=0.1 # Float or None
                    ),
    ],
}

multi_input_conv_3D_config = {
    'conv_units': [
                    dict(
                        in_channels=1,
                        out_channels=16,
                        kernel_size=(3, 3, 3),
                        stride=2, # Makes a big difference in training speeds
                        dropout=0.1 # Float or None
                    ),
    ],
}
dense_config = {
    'dense_units': [100, 50],
    'initializer': None,
    'bias_init': None,
    'norm': None,
    'dropout': 0.5,  # Single value or List
}

conv_1D = ConvNet(
    name='conv_1D',
    input_networks=None,
    in_dim=(1, 28),
    config=conv_1D_config,
)
conv_2D = ConvNet(
    name='conv_2D',
    input_networks=None,
    in_dim=(1, 28, 28),
    config=conv_2D_config
)
conv_3D = ConvNet(
    name='conv_3D',
    input_networks=None,
    in_dim=(1, 28, 28, 28),
    config=conv_3D_config,
)

dense_model = DenseNet(
    name='dense_model',
    input_networks=[conv_2D, conv_1D],
    config=dense_config
)

multi_input_conv_3D = ConvNet(
    name='multi_input_conv_3D',
    input_networks=[conv_1D, dense_model, conv_2D, conv_3D],
    config=multi_input_conv_3D_config,
    num_classes=10
)


multi_dense = [
    (val_loader.dataset, True, False),
    (TensorDataset(torch.ones([10000, *conv_1D.in_dim])), True, False)
]

m = MultiDataset(multi_dense)

x = [
        (TensorDataset(torch.ones([10000, *conv_1D.in_dim])), True, False),
        m,
        (val_loader.dataset, True, True),
        (TensorDataset(torch.ones([10000, *conv_3D.in_dim])), True, False),
    ]

multi_dataset = MultiDataset(x)

train_multi = torch.utils.data.Subset(
    multi_dataset, range(len(multi_dataset)//2))
val_multi = torch.utils.data.Subset(
    multi_dataset, range(len(multi_dataset)//2, len(multi_dataset)))

train_loader_multi = DataLoader(train_multi, batch_size=100)
val_loader_multi = DataLoader(val_multi, batch_size=100)

multi_input_conv_3D.fit(train_loader_multi, val_loader_multi, 3, plot=True)

multi_input_conv_3D.run_test(val_loader_multi, plot=True)
multi_input_conv_3D.save_model()
