from vulcanai import datasets
from vulcanai.models import ConvNet, DenseNet
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
import numpy as np

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


conv_2D = ConvNet(
    name='conv_2D',
    in_dim=(1, 28, 28),
    config=conv_2D_config,
    num_classes=1,
    criter_spec=torch.nn.MSELoss()
)

test_input = torch.rand(size=[10, 1, 28, 28])
test_output = torch.rand(size=[10, 1])

test_dataloader = DataLoader(TensorDataset(test_input, test_output))

conv_2D.fit(
    test_dataloader,
test_dataloader,
    epochs=3,
    plot=False,
    save_path="."
)

conv_2D.run_test(test_dataloader)
res = conv_2D.forward_pass(test_dataloader, convert_to_class=True, transform_callable=np.round, decimals=3)

print(res)