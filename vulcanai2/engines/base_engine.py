__author__ = 'Caitrin'
"""

The engine deals with training, evaluating, saving and checkpointing. Most of the code will be written in this class, but can be overridden where necessary

"""


"""

- Need to adjust learning rate based on the number of epochs

"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from datetime import datetime
import time
from tqdm import tqdm

class Engine(object):

    def train(self, network, epochs, train_loader, test_loader, criterion, optimizer,
              scheduler=None, use_gpu=False):

        print('\nTraining in progress...Time now: {}\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

        try:

            if use_gpu:
                network = network.cuda()
                criterion = criterion.cuda()

            for epoch in range(epochs):
                epoch_time = time.time()
                print("--> Epoch: {}/{}".format(
                    epoch,
                    epochs - 1
                ))
                for mode in ['train', 'val']:
                    if mode == 'train':
                        network.train()  # Set model to training mode

                        loader = tqdm(train_loader, total=len(train_loader))

                        for batch_idx, (data, target) in enumerate(loader):

                            data = Variable(data.float())
                            target = Variable(target)

                            if use_gpu:
                                data = data.cuda()
                                target = target.cuda()

                            # Forward + Backward + Optimize
                            optimizer.zero_grad()
                            output = network(data)
                            loss = nn.CrossEntropyLoss()(output, target)
                            loss.backward()
                            optimizer.step()
                        print('\tTrain set: Loss: {:.6f}'.format(loss.item()))

                    else:
                        network.eval()  # Set model to evaluate mode
                        test_loss = 0
                        correct = 0
                        with torch.no_grad():
                            for data, target in test_loader:

                                data = Variable(data.float())
                                target = Variable(target)

                                if torch.cuda.is_available():
                                    data = data.cuda()
                                    target = target.cuda()

                                output = network(data)
                                test_loss += nn.CrossEntropyLoss()(output, target)
                                _, pred = torch.max(output.data, 1)
                                #pred = output.max(1)[1] # get the index of the max log-probability
                                correct += (pred == target).sum()

                        test_loss /= len(test_loader.dataset)
                        print('\n\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                            test_loss, correct, len(test_loader.dataset),
                            100. * correct / len(test_loader.dataset)))
        except KeyboardInterrupt:
            print("\n\n**********Training stopped prematurely.**********\n\n")
        finally:
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))