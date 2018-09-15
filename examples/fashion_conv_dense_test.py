import sys
sys.path.append('../')
from vulcanai2 import models, datasets

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from torchviz import make_dot

import numpy as np
import json
from collections import OrderedDict

sys.path.append('../')
normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])


data_path = r'C:\WORK\Aifred\Vulcan2\master\Vulcan2\data'
#data_path = r'/Users/robertfratila/Code/Aifred_Health/Vulcan2/data'

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
                        padding=0
                    ),
                    dict(
                        in_ch=16,
                        out_ch=32,
                        k_size=(5, 5),
                        stride=1,
                        padding=0
                    ),
                    dict(
                        in_ch=32,
                        out_ch=64,
                        k_size=(5, 5),
                        stride=1,
                        padding=0
                        )
    ],
}
dense_net_config = {
    'dense_units': [100],
    'dropouts': [0.3],
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

#model1.fit(train_loader, val_loader, 10)

#print(model1.run_test(np.expand_dims(val_dataset.test_data, axis=1), np.expand_dims(val_dataset.test_labels, axis=1)))

params = model1.get_output_shapes((1,28,28))
for k, v in params.items() :
    print('{}:'.format(k))
    if isinstance(v, OrderedDict):
        for k2, v2 in v.items():
            print('\t {}: {}'.format(k2, v2))
