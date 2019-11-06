# -*- coding: utf-8 -*-

from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import codecs
import torchvision.transforms as transforms
import urllib


# FROM https://raw.githubusercontent.com/mayurbhangale/
# fashion-mnist-pytorch/master/fashion.py
class FashionData(data.Dataset):
    """'MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Parameters:
        root (string): Root directory of dataset where
        ``processed/training.pt`` and ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the
        internet and puts it in root directory. If dataset is already
        downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an
        PIL image and returns a transformed version. E.g,
        ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that
        takes in the target and transforms it.
    """
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,),
                                                                 (0.3081,))])

        if download:
            self.download()

        if not self._check_exists():
            print("downloading")
            self.download()

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(root, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(os.path.join(
                root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder,
                                           self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder,
                                        self.test_file))

    # TODO: Need to fix. File not found error before it downloads
    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder
        already."""
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            response = urllib.request.urlopen(url)
            #compressed_file = io.BytesIO(response.read())
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(fileobj=response) as zip_f:
                out_f.write(zip_f.read())
            #os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder,
                                         'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder,
                                         'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder,
                                         't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder,
                                         't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder,
                               self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder,
                               self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

def parse_byte(b):
    if isinstance(b, str):
        return ord(b)
    return b

def read_label_file(path):
    with open(path, 'rb') as f:
        data_in = f.read()
        assert get_int(data_in[:4]) == 2049
        length = get_int(data_in[4:8])
        labels = [parse_byte(b) for b in data_in[8:]]
        assert len(labels) == length
        return torch.LongTensor(labels)

def read_image_file(path):
    with open(path, 'rb') as f:
        data_in = f.read()
        assert get_int(data_in[:4]) == 2051
        length = get_int(data_in[4:8])
        num_rows = get_int(data_in[8:12])
        num_cols = get_int(data_in[12:16])
        images = []
        idx = 16
        for l in range(length):
            img = []
            images.append(img)
            for r in range(num_rows):
                row = []
                img.append(row)
                for c in range(num_cols):
                    row.append(parse_byte(data_in[idx]))
                    idx += 1
        assert len(images) == length
        return torch.ByteTensor(images).view(-1, 28, 28)
