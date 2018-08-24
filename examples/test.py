import sys
sys.path.append('../')
from vulcanai2 import models, datasets

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


sys.path.append('../')
normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.FashionData(root='C:\WORK\Aifred\Vulcan2\master\Vulcan2\data', 
                            train=True, 
                            transform=transform,
                            download=True
                           )

val_dataset = datasets.FashionData(root='C:\WORK\Aifred\Vulcan2\master\Vulcan2\data', 
                            train=False, 
                            transform=transform,
                           )


batch_size = 100
n_iters = 5500

train_loader = DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size,            
                                           shuffle=True)

val_loader = DataLoader(dataset=val_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)


network_conv_config = {
    'mode': 'conv',
    'units': ([ [1, 16, (5, 5), 2, 0],
                [16, 32, (5, 5), 1, 0],
                [32, 64, (5, 5), 1, 0] ],
              [ 100 ]),
    'dropouts': [0.2],
}

model = models.ConvNet(
    name='conv_net_test',
    dimensions=(1, 28, 28),
    config=network_conv_config,
    num_classes=10
)

model.fit(train_loader, val_loader, 10)