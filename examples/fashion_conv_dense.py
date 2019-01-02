"""Simple Convolution and fully connected blocks example."""
from vulcanai import datasets
from vulcanai.models import ConvNet, DenseNet

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x/255.0 for x in [63.0, 62.1, 66.7]])

transform = transforms.Compose([transforms.ToTensor(),
                                normalize])


data_path = "../data"
train_dataset = datasets.FashionData(root=data_path,
                                     train=True,
                                     transform=transform,
                                     download=True)

val_dataset = datasets.FashionData(root=data_path,
                                   train=False,
                                   transform=transform)


batch_size = 100

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
    num_classes=10
)

dense_model.fit(
    train_loader,
    val_loader,
    epochs=3,
    # plot=True
)
dense_model.run_test(val_loader, plot=True)
dense_model.save_model()
