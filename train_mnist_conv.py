__author__ = 'Caitrin'
import numpy as np
from vulcanai2.dataloaders import fashion
from datetime import datetime
import time
from tqdm import tqdm
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


criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

conv_net = Network(
    name='conv_test',
    dimensions=[None, 1] + list(train_images.shape[1:]),
    input_var=input_var,
    y=y,
    config=network_conv_config,
    input_network=None,
    num_classes=None)


train_images = np.expand_dims(train_images, axis=1)
test_images = np.expand_dims(test_images, axis=1)
# # Use to load model from disk
# # dense_net = Network.load_model('models/20170704194033_3_dense_test.network')
conv_net.train(
    epochs=200,
    train_x=train_images[:50000],
    train_y=train_labels[:50000],
    val_x=train_images[50000:60000],
    val_y=train_labels[50000:60000],
    batch_ratio=0.05,
    plot=False
)

#save record
#save model