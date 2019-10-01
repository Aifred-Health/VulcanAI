"""Simple Convolution and fully connected blocks cross validation example."""
from vulcanai import datasets
from vulcanai.models import ConvNet, DenseNet
from vulcanai.models.metrics import Metrics

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# prepare the data
normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x/255.0 for x in [63.0, 62.1, 66.7]])

transform = transforms.Compose([transforms.ToTensor(),
                                normalize])


data_path = "../data"
dataset = datasets.FashionData(root=data_path,
                                     train=True,
                                     transform=transform,
                                     download=True)

batch_size = 100

data_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True)



# define neural network - 3 2D conv layers followed by a dense layer
conv_2D_config = {
    'conv_units': [
                    dict(
                        in_channels=1,
                        out_channels=16,
                        kernel_size=(5, 5),
                        stride=2,
                        dropout=0.1
                    ),
                    dict(
                        in_channels=16,
                        out_channels=32,
                        kernel_size=(5, 5),
                        dropout=0.1
                    ),
                    dict(
                        in_channels=32,
                        out_channels=64,
                        kernel_size=(5, 5),
                        pool_size=2,
                        dropout=0.1
                        )
    ],
}

dense_config = {
    'dense_units': [100, 50],
    'dropout': 0.5,  # Single value or List
}

conv_2D = ConvNet(
    name='conv_2D',
    in_dim=(1, 28, 28),
    config=conv_2D_config
)

dense_model = DenseNet(
    name='dense_model',
    input_networks=conv_2D,
    config=dense_config,
    num_classes=10,
    early_stopping="best_validation_error",
    early_stopping_patience=2
)


# cross validate on 5 folds training each fold for 2 epochs
m = Metrics()

m.cross_validate(dense_model, data_loader, 5, 2)
