import sys
sys.path.append('../')
from vulcanai2 import models, datasets, plotters

import pickle
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from torchviz import make_dot

import numpy as np
import json
from collections import OrderedDict
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


conv_net_config = {
    'conv_units': [
                    dict(
                        in_ch=1,
                        out_ch=16,
                        k_size=(5, 5),
                        stride=2,
                        padding=0,
                        initializer=None,
                        bias_init=None, # None or value
                        norm=None,
                        pool_size=None,
                        dropout=0.1 # Float or None
                    ),
                    dict(
                        in_ch=16,
                        out_ch=32,
                        k_size=(5, 5),
                        stride=1,
                        padding=0,
                        initializer=None,
                        bias_init=None, # None or value
                        norm=None,
                        pool_size=None,
                        dropout=0.1 # Float or None
                    ),
                    dict(
                        in_ch=32,
                        out_ch=64,
                        k_size=(5, 5),
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
    'dense_units': [100],
    'initializers':[None],
    'bias_inits':[0], # List of None or values
    'norms':[None],
    'dropouts': [0.3],  # List of Float or None
}

model = models.ConvNet(
    name='conv_net_test',
    input_network=None,
    dimensions=(1, 28, 28),
    config=conv_net_config
)

model1 = models.DenseNet(
    name='dense_net_test',
    input_network=model,
    dimensions=model.conv_flat_dim,
    config=dense_net_config,
    num_classes=10
)
print(model1)
#model1.fit(train_loader, val_loader, 10)
model1.fit(train_loader, val_loader, 2, plot=True)

model1.save_model()

#model2 = models.DenseNet.load_ensemble("/home/caitrin/Vulcan2/Vulcan2/examples/2018-10-04_19:12:36/dense_net_test")

#model2.fit(train_loader, val_loader, 4, plot=True)

# TODO: need to revisit this to be able to plot after training, interactive plotting is messing up
#plotters.visualization.display_record(record=model1.record, interactive=False)
#plt.show()

#model1.print_model_structure()

