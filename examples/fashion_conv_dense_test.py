import sys
sys.path.append('../')
from vulcanai2 import models, datasets, plotters
from vulcanai2.models.cnn import ConvNet
from vulcanai2.models.dnn import DenseNet
from vulcanai2.plotters.visualization import compute_saliency_map, display_saliency_overlay, display_receptive_fields

import pickle
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
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
    input_networks=None,
    dimensions=(1, 28, 28),
    config=conv_net_config,
)

dense_model = DenseNet(
    name='conv_net_test',
    input_networks=None,
    dimensions=[784],
    config={
    'dense_units': [500, 100],
    'initializer': None,
    'bias_init': None,
    'norm': None,
    'dropout': 0.5,  # Single value or List
    },
)

model1 = DenseNet(
    name='dense_net_test',
    input_networks=[model, dense_model],
    dimensions=[model.conv_flat_dim, dense_model.out_dim],
    config=dense_net_config,
    num_classes=10
)
# od1 = model.get_output_shapes((1, 28, 28))

# print(list(od1))
# for k,v in model.get_output_shapes((1, 28, 28)).items():
#     print('{}:'.format(k))
#     if isinstance(v, odict):
#         for k2, v2 in v.items():
#             print('\t {}: {}'.format(k2, v2))
# print(model1.get_output_shapes())
# d = DenseNet(
#             name='Test_DenseNet_class',
#             dimensions=(200),
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
# NOTE: in_dim must be list of tuples or ints | out_dim must represent for single output so just tuple or int
print(model.in_dim, model.out_dim, dense_model.in_dim, dense_model.out_dim, model1.in_dim, model1.out_dim)
print(model1.get_output_shapes(input_size = [(1,28,28), 784])) #TODO: Ensure to make this work without specifying the input size
model1.fit(train_loader, val_loader, 2, plot=False)

model1.save_model()

#model2 = models.DenseNet.load_ensemble("/home/caitrin/Vulcan2/Vulcan2/examples/2018-10-04_19:12:36/dense_net_test")

#model2.fit(train_loader, val_loader, 4, plot=True)

# To test saliency map generation
# x = train_loader.dataset.train_data[:5].float().unsqueeze(dim=1) #np.expand_dims(train_loader.dataset[:5][0], axis=0)
# y = train_loader.dataset.train_labels[:5]
# sal_map = compute_saliency_map(model1, x, y)
# display_saliency_overlay(train_loader.dataset.train_data[0], sal_map[0])

# TODO: need to revisit this to be able to plot after training, interactive plotting is messing up
#plotters.visualization.display_record(record=model1.record, interactive=False)
#plt.show()

#model1.print_model_structure()

