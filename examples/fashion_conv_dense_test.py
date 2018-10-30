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


conv_net_config_small = {
    'conv_units': [
                    dict(
                        in_channels=1,
                        out_channels=16,
                        kernel_size=(5),
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
                        kernel_size=(5),
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
                        kernel_size=(5),
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
conv_net_config_big = {
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
conv_net_config_very_big = {
    'conv_units': [
                    dict(
                        in_channels=1,
                        out_channels=16,
                        kernel_size=(5, 5, 5),
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
                        out_channels=16,
                        kernel_size=(5, 5, 5),
                        stride=1, # Makes a big difference in training speeds
                        padding=0,
                        initializer=None,
                        bias_init=None, # None or value
                        norm=None,
                        pool_size=None,
                        dropout=0.1 # Float or None
                    ),
                    dict(
                        in_channels=16,
                        out_channels=64,
                        kernel_size=(5, 5, 5),
                        stride=1, # Makes a big difference in training speeds
                        padding=0,
                        initializer=None,
                        bias_init=None, # None or value
                        norm=None,
                        pool_size=None,
                        dropout=0.1 # Float or None
                    ),
    ],
}

conv_net_config_very_very_big = {
    'conv_units': [
                    dict(
                        in_channels=1,
                        out_channels=16,
                        kernel_size=(3, 3, 3),
                        stride=2, # Makes a big difference in training speeds
                        padding=0,
                        initializer=None,
                        bias_init=None, # None or value
                        norm=None,
                        pool_size=None,
                        dropout=0.1 # Float or None
                    ),
    ],
}
dense_net_config = {
    'dense_units': [100, 50],
    'initializer': None,
    'bias_init': None,
    'norm': None,
    'dropout': 0.5,  # Single value or List
}

conv_small = ConvNet(
    name='conv_net_small',
    input_networks=None,
    in_dim=(1, 28),
    config=conv_net_config_small,
)
conv_big = ConvNet(
    name='conv_net_big',
    input_networks=None,
    in_dim=(1, 28, 28),
    config=conv_net_config_big,
    # num_classes=10
)
conv_very_big = ConvNet(
    name='conv_net_very_big',
    input_networks=None,
    in_dim=(1, 28, 28, 28),
    config=conv_net_config_very_big,
)

dense_model = DenseNet(
    name='conv_net_test',
    input_networks=[conv_big, conv_small],
    in_dim=150,
    config=dense_net_config,
    # num_classes=10
)

# dense_output = dense_model(torch.ones([5,*conv_big.in_dim]))

x = train_loader.dataset.train_data[:5].float().unsqueeze(dim=1) #np.expand_dims(train_loader.dataset[:5][0], axis=0)
y = train_loader.dataset.train_labels[:5]

# x = [
#         torch.ones([5,*conv_big.in_dim]),
#         torch.ones([5,*conv_small.in_dim])]
multi_dense = [
    (val_loader.dataset, True, False),
    (TensorDataset(torch.ones([10000,*conv_small.in_dim])), True, False)
]

m = MultiDataset(multi_dense)

x = [
        (TensorDataset(torch.ones([10000,*conv_small.in_dim])), True, False),
        m,
        (val_loader.dataset, True, True),
        (TensorDataset(torch.ones([10000,*conv_very_big.in_dim])), True, False),
    ]

multi = DataLoader(MultiDataset(x), batch_size=100)

# sal_map = compute_saliency_map(model1, x, y)

# display_saliency_overlay(train_loader.dataset.train_data[0], sal_map[0][0])

model1 = ConvNet(
    name='conv_net_test_multi_input',
    input_networks=[conv_small, dense_model, conv_big, conv_very_big],
    #in_dim=[conv_small.out_dim, dense_model.out_dim, conv_big.out_dim, conv_very_big.out_dim],
    config=conv_net_config_very_very_big,
    num_classes=10
)
# a = model1.forward_pass(multi)
model1.fit(multi, multi, 30, plot=True)
import pudb; pu.db
dense_model.fit(train_loader, val_loader, 2)
import pudb; pu.db
# snap = SnapshotNet("snap", dense_model, 3)
#conv_big.fit(train_loader, val_loader, 3, plot=True)
# snap.save_model()
#import pudb; pu.db
#model1.save_model()
very_very_big_conv_output = model1(
    [
        torch.ones([5,*model1.input_networks[0].in_dim]),
        torch.ones([5,*model1.input_networks[1].in_dim]),
        torch.ones([5,*model1.input_networks[2].in_dim]),
        torch.ones([5,*model1.input_networks[3].in_dim])
    ])

# print(model1.get_output_shapes())
# d = DenseNet(
#             name='Test_DenseNet_class',
#             in_dim=(200),
#             config={
#                 'dense_units': [100],
#                 'dropouts': [0.3],
#             },
#             num_classes=3
#         )
# rf = display_receptive_fields(d)

# test_input_1B = np.ones([1, d.in_dim], dtype=np.float32)
# sal_map_1B = compute_saliency_map(
#             d,
#             test_input_1B, torch.tensor([2]))


#model1.fit(train_loader, val_loader, 10)


model1.fit(train_loader, val_loader, 2, plot=False)

# test_input_1B = np.ones([1, d.in_dim], dtype=np.float32)
# sal_map_1B = compute_saliency_map(
#             d,
#             test_input_1B, torch.tensor([2]))


#model1.fit(train_loader, val_loader, 10)
model1.fit(train_loader, val_loader, 2, plot=False)

#model1.save_model()

#model2 = models.DenseNet.load_ensemble("/home/caitrin/Vulcan2/Vulcan2/examples/2018-10-04_19:12:36/dense_net_test")

#model2.fit(train_loader, val_loader, 4, plot=True)

# To test saliency map generation
# model1.run_test(val_loader, plot=True)

# f_pass = model1.forward_pass(val_loader, convert_to_class=True)

# cm = get_confusion_matrix(
#     model1.forward_pass(val_loader, convert_to_class=True),
#     val_loader.dataset.test_labels)
# display_confusion_matrix(cm, ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle"])

x = train_loader.dataset.train_data[:5].float().unsqueeze(dim=1) #np.expand_dims(train_loader.dataset[:5][0], axis=0)
y = train_loader.dataset.train_labels[:5]
sal_map = compute_saliency_map(dense_model, x, y)
display_saliency_overlay(train_loader.dataset.train_data[0], sal_map[0])

se = SnapshotNet("snap", dense_model, 3)

# Does it make more sense to pass the total # of epochs
# or just how many each model should train for?

snap.fit(train_loader, val_loader, 3, plot=True)
snap.run_test(val_loader, plot=True)

snap.save_model()


# se = SnapshotNet.load_model('saved_models/snap_2018-10-17_00-06-17')
preds = snap.forward_pass(val_loader, convert_to_class=True)


# TODO: need to revisit this to be able to plot after training, interactive plotting is messing up
#plotters.visualization.display_record(record=model1.record, interactive=False)
#plt.show()

#model1.print_model_structure()

