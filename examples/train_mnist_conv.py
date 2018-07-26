__author__ = 'Caitrin'
import numpy as np

imagenet_data = torchvision.datasets.ImageFolder('path/to/imagenet_root/')
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=args.nThreads)

from sklearn.utils import shuffle

train_images, train_labels = shuffle(train_images, train_labels, random_state=0)


label_map = {
    '0': 'T-shirt/top',
    '1': 'Trouser',
    '2': 'Pullover',
    '3': 'Dress',
    '4': 'Coat',
    '5': 'Sandal',
    '6': 'Shirt',
    '7': 'Sneaker',
    '8': 'Bag',
    '9': 'Ankle boot'
}


train_labels = get_one_hot(train_labels)
test_labels = get_one_hot(test_labels)

train_images = np.reshape(train_images, (train_images.shape[0], 28, 28))
test_images = np.reshape(test_images, (test_images.shape[0], 28, 28))


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

dense_net.save_record()

run_test(dense_net, test_x=test_images, test_y=test_labels)
dense_net.save_model()