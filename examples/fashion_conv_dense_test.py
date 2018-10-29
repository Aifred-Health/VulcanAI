import sys
sys.path.append('../')
from vulcanai2 import models, datasets, plotters
from vulcanai2.models import ConvNet, DenseNet, SnapshotNet
# from vulcanai2.models.cnn import ConvNet
# from vulcanai2.models.dnn import DenseNet
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
from collections import OrderedDict
import matplotlib.pyplot as plt

from vulcanai2.models.metrics import Metrics

sys.path.append('../')
normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])


#data_path = r'C:\WORK\Aifred\Vulcan2\master\Vulcan2\data'
#data_path = r'/Users/robertfratila/Code/Aifred_Health/Vulcan2/data'
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


conv_net_config = {
    'conv_units': [
                    dict(
                        in_channels=1,
                        out_channels=16,
                        kernel_size=(5, 5),
                        stride=2, # Makes a big difference in training speeds
                        padding=0,
                        initializer=None,
                        bias_init=None, # None or value
                        norm=None,
                        pool_size=None,
                        dropout=0.1 # Float or None
                    ),
                    dict(
                        in_channels=16,
                        out_channels=32,
                        kernel_size=(5, 5),
                        stride=1,
                        padding=0,
                        initializer=None,
                        bias_init=None, # None or value
                        norm=None,
                        pool_size=None,
                        dropout=0.1 # Float or None
                    ),
                    dict(
                        in_channels=32,
                        out_channels=64,
                        kernel_size=(5, 5),
                        stride=1,
                        padding=0,
                        initializer=None,
                        bias_init=None, # None or value
                        norm=None,
                        pool_size=2,
                        dropout=0.1 # Float or None
                        )
    ],
}
dense_net_config = {
    'dense_units': [100, 50],
    'initializer': None,
    'bias_init': None,
    'norm': None,
    'dropout': 0.5,  # Single value or List
}

model = ConvNet(
    name='conv_net_test',
    input_network=None,
    dimensions=(1, 28, 28),
    config=conv_net_config,
)

model1 = DenseNet(
    name='dense_net_test',
    input_network=model,
    dimensions=model.conv_flat_dim,
    config=dense_net_config,
    num_classes=10
)

#model1.fit(train_loader, val_loader, 2, plot=True)
#model1.k_fold_cross_validation(train_loader, 5, 2, plot=False)

# model1.save_model()

#model2 = models.DenseNet.load_ensemble("/home/caitrin/Vulcan2/Vulcan2/examples/2018-10-04_19:12:36/dense_net_test")

#model2.fit(train_loader, val_loader, 4, plot=True)

# To test saliency map generation
# model1.run_test(val_loader, plot=True)

cnn_class = ConvNet(
        name='Test_ConvNet_class',
        dimensions=(1, 28, 28),
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
        num_classes=10
    )

k = 2
epochs = 2

averaged_results = cnn_class.cross_validate(train_loader, k, epochs, average_results=True)
#all_results = cnn_class.cross_validate(test_dataloader, k, epochs, average_results=False)

print(averaged_results)
#print(all_results)
#assert len(averaged_results.values()[0]) == 1
#assert len(all_results.values()[0]) == k

#print(model1.cross_validate(train_loader, 5, 2, plot=False, average_results=False))

# f_pass = model1.forward_pass(val_loader, convert_to_class=True)

# cm = get_confusion_matrix(
#     model1.forward_pass(val_loader, convert_to_class=True),
#     val_loader.dataset.test_labels)
# display_confusion_matrix(cm, ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle"])

# x = train_loader.dataset.train_data[:5].float().unsqueeze(dim=1) #np.expand_dims(train_loader.dataset[:5][0], axis=0)
# y = train_loader.dataset.train_labels[:5]
# sal_map = compute_saliency_map(model1, x, y)
# display_saliency_overlay(train_loader.dataset.train_data[0], sal_map[0])

#se = SnapshotNet("snap", model1, 3)

# Does it make more sense to pass the total # of epochs
# or just how many each model should train for?

#se.fit(train_loader, val_loader, 3, plot=True)
#se.run_test(val_loader, plot=True)

#se.save_model()

# se = SnapshotNet.load_model('saved_models/snap_2018-10-17_00-06-17')
#preds = se.forward_pass(val_loader, convert_to_class=True)


# TODO: need to revisit this to be able to plot after training, interactive plotting is messing up
#plotters.visualization.display_record(record=model1.record, interactive=False)
#plt.show()

#model1.print_model_structure()

