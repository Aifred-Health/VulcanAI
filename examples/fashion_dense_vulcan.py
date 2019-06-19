"""Simple Convolution and fully connected blocks example."""
from vulcanai import datasets
from vulcanai.models import ConvNet, DenseNet
from torch import nn
import torchvision.transforms as transforms
import tensorflow as tf
from torch.utils.data import TensorDataset, DataLoader
import torch
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
# Further break training data into train / validation sets (# put 5000 into validation set and keep remaining 55,000 for train)
(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

#Reshape input data from (28, 28) to (28, 28, 1)
w, h = 28, 28
x_train = x_train.reshape(x_train.shape[0], 1, w, h)
x_valid = x_valid.reshape(x_valid.shape[0], 1, w, h)
x_test = x_test.reshape(x_test.shape[0], 1, w, h)

#One-hot encode the labels
# y_train = tf.keras.utils.to_categorical(y_train, 10)
# y_valid = tf.keras.utils.to_categorical(y_valid, 10)
# y_test = tf.keras.utils.to_categorical(y_test, 10)

print(x_train.shape, "SHAPE")

batch_size = 100

train_dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
val_dataset = TensorDataset(torch.from_numpy(x_valid).float(), torch.from_numpy(y_valid).long())


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=batch_size,
                        shuffle=False)

conv_2D_config = {
    'conv_units': [
                    dict(
                        in_channels=1,
                        out_channels=16,
                        kernel_size=(5, 5),
                        stride=2,
                        pool_size=2,
                        dropout=0.1
                    ),
                    dict(
                        in_channels=16,
                        out_channels=32,
                        kernel_size=(5, 5),
                        pool_size=2,
                        dropout=0.1
                        )
    ],
}

conv_2D = ConvNet(
    name='conv_2D',
    in_dim=(1, 28, 28),
    config=conv_2D_config,
    num_classes=10
)

conv_2D.fit(
    train_loader,
    val_loader,
    epochs=1,
)
conv_2D.run_test(val_loader, plot=True, save_path=".")
conv_2D.save_model()
