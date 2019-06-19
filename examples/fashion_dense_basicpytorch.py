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

epochs = 1

LR = 0.001

train_dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
val_dataset = TensorDataset(torch.from_numpy(x_valid).float(), torch.from_numpy(y_valid).long())


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=batch_size,
                        shuffle=False)

class My_Model(nn.Module):
    def __init__(self, num_of_class):
        super(My_Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5,5), stride=2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(kernel_size=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(5,5)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(kernel_size=2))
        self.fc = nn.Linear(32, num_of_class)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1) #flatten
        out = self.fc(out)
        return out

def train():
    model = My_Model(10)
    optimizer = torch.optim.Adam(model.parameters(), lr = LR)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        for batch_id, (image, label) in enumerate(train_loader):
            output = model(image)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Loss :{:.4f} Epoch[{}/{}]'.format(loss.item(), epoch, epochs))
    return model

def test(model):
    with torch.no_grad():
        correct = 0
        total = 0
        for image, label in val_loader:
            outputs = model(image)
            m = torch.nn.Softmax(dim=1)
            #outputs = m(outputs)
            predicted = torch.argmax(outputs ,dim=1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))

if __name__ == '__main__':
    model = train()
    test(model)