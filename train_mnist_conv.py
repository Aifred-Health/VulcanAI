__author__ = 'Caitrin'
import numpy as np
from vulcanai2.dataloaders import fashion
from datetime import datetime
import time
from tqdm import tqdm
from vulcanai2.models import cnn
from torch.utils.data import DataLoader #WE CAN DO A CUSTOM IMPORT HERE


train_dataset = fashion.fashion(root='data',
                            train=True,
                            download=True
                           )

test_dataset = fashion.fashion(root='data',
                            train=False,
                           )

batch_size = 100
n_iters = 5500
num_epochs = int(n_iters / (len(train_dataset) / batch_size))

train_loader = DataLoader(dataset=train_dataset,
                        batch_size=batch_size,
                        shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                        batch_size=batch_size,
                        shuffle=False)

network_conv_config = {
    'mode': 'conv',
    'filters': [16, 32],
    'filter_size': [[5, 5], [5, 5]],
    'stride': [[1, 1], [1, 1]],
    'pool': {
        'mode': 'average_exc_pad',
        'stride': [[2, 2], [2, 2]]
    }
}

input_var = None
y= None

conv_net = cnn(
    name='conv_test',
    dimensions=[None, 1] + list(train_images.shape[1:]),
    input_var=input_var,
    y=y,
    config=network_conv_config,
    input_network=None,
    num_classes=None)

# # Use to load model from disk
# # dense_net = Network.load_model('models/20170704194033_3_dense_test.network')
conv_net.train(
    network,
    epochs,
    train_loader, #TODO: we probably want to give a ratio to split this for validation.....? maybe we don't always want to do this?
    test_loader,
    criterion,
    optimizer,
    change_rate=None,
    use_gpu=False
)

#save record
#save model