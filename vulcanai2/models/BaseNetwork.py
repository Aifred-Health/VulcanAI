__author__ = 'Caitrin'

import abc
import torch
import torch.nn as nn
import datetime
import torch.nn.modules.activation as activations #TODO: want to call it activation but namespace, so what to do best?
from torch import optim
import torch.nn.modules.loss as loss
import layers #TODO: I don't know why pycharm keeps rejecting this?
import tqdm
import time
from torch.autograd import Variable

class BaseNetwork():

    #TODO: this should be the same for every model, given that you pass a config??
    #TODO: come up with an alternate way to validate config
    #TODO: deal with stopping rules
    #TODO: do you have to call any nn.module methods?? do you even actually want to subclass at this point if you have layers that subclass nn.module??
    #TODO: will need to create a wrapper class to use non default keworded parameters for all torch objects
    #TODO: reorder these?
    def __init__(self, name, dimensions, config, save_path=None, input_network=None, num_classes=None, activation=activations.ReLU(), pred_activation=activations.Softmax(), optimizer=optim.Adam, learning_rate=0.001, lr_scheduler=None, stopping_rule='best_validation_error', criterion=None):
        """
        :param name:
        :param dimensions:
        :param config:
        :param save_path:
        :param input_network:
        :param num_classes:
        :param activation:
        :param pred_activation:
        :param optimizer:
        :param learning_rate:
        :param lr_scheduler:
        :param stopping_rule:
        :param criterion:
        :return:
        """
        self._name = name
        self._dimensions = dimensions
        self._config = config

        self._save_path = save_path

        self._input_network = input_network #TODO: change and check type here?
        self._num_classes = num_classes
        self._activation = activation
        self._pred_activation = pred_activation
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._lr_scheduler = lr_scheduler
        self._stopping_rule = stopping_rule
        self._criterion = criterion
        self._network = None
        self._create_network()


    #TODO: where to do typechecking... just let everything fail?
    #TODO: add on additional if you want to be able to re-create a network?

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def save_path(self):
        return self._save_path

    @save_path.setter
    def save_path(self, value):
        if not value:
            self.save_path = "{}_{date:%Y-%m-%d_%H:%M:%S}/".format(self.name, date=datetime.datetime.now())
        else:
            self._save_path = value
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        #TODO: check type?
        self._optimizer = value

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value

    @property
    def lr_scheduler(self):
        return self._lr_scheduler

    @lr_scheduler.setter
    def lr_scheduler(self, value):
        self._lr_scheduler = value

    @property
    def stopping_rule(self):
        return self._stopping_rule

    @stopping_rule.setter
    def stopping_rule(self, value):
        self._stopping_rule = value

    @property
    def criterion(self):
        return self._criterion

    @criterion.setter
    def criterion(self, value):
        if not value and not self._num_classes:
            self._criterion = loss.MSELoss
        elif not value:
            self._criterion = loss.CrossEntropyLoss
        else:
            self._criterion = value


    # #TODO: figure out how this works in conjunction with optimizer
    # #TODO: fix the fact that you copy pasted this
    # def cuda(self, device_id=None):
    #     """Moves all model parameters and buffers to the GPU.
    #     Arguments:
    #         device_id (int, optional): if specified, all parameters will be
    #             copied to that device
    #     """
    #     self.is_cuda = True
    #     return self._apply(lambda t: t.cuda(device_id))
    #
    # def cpu(self):
    #     """Moves all model parameters and buffers to the CPU."""
    #     self.is_cuda = False
    #     return self._apply(lambda t: t.cpu())

    #TODO: I think this needs to take into account passing through networks
    #TODO: make this isn't resetting things when you have mulitple networks
    @abc.abstractmethod
    def _create_network(self):
        pass

    # #TODO: deal with the fact that you copied this
    # #TODO: figure out if you really need this?
    # def prepare_batch(self, batch):
    #     if self.is_cuda:
    #         batch = self.cuda_tf()(batch)
    #     if self.mode == 'eval':
    #         batch = self.detach_tf()(batch)
    #     return batch

    def get_all_layers(self):
        layers = []
        for l_name, l in self.input_network['network'].network.named_children():
            if isinstance(l, nn.Sequential):
                for subl_name, subl in l.named_children():
                    layers.append(subl)
            else:
                for param in l.parameters():
                    self.input_dimensions= param.size(0)


    #TODO: do you really want this here....?
    def create_classification_layer(self):
        """
        Create a classification layer. Normally used as the last layer.
        Args:
            network: network you want to append a classification to
            num_classes: how many classes you want to predict
            nonlinearity: nonlinearity to use as a string (see DenseLayer)
        Returns: the classification layer appended to all previous layers
        """
        print('\tOutput Layer:')
        layer_name ="classification_layer"
        layer = layers.DenseUnit(
                          in_channels=self.input_dim,
                          out_channels=self.num_classes,
                          bias=True,
                          activation=self.pred_nonlinearity,
                          norm=None)
        self.network.add_module(layer_name, layer)
        self.layers.append(layer)
        print('\t\t{}'.format(layer))

    #TODO: this won't work for all of them...
    def _initialize_optimizer(self):
        return self.optimizer(self._network.parameters, lr = self._learning_rate)

    def _initialize_scheduler(self, optimizer):
        return self._scheduler(optimizer)

    #TODO: use_gpu should probably go somewhere else in the future...
    def fit(self, train_loader, val_loader, epochs):

        #need to do this before constructing an optimizer
        if torch.cuda.is_available():
            network = self._network.cuda()
            criterion = self._criterion.cuda()

        optimizer = self._initialize_optimizer()

        if self._scheduler:
            scheduler = self._initialize_scheduler(self, optimizer)

        try:
            for epoch in range(epochs):
                epoch_time = time.time()
                print("--> Epoch: {}/{}".format(
                    epoch,
                    epochs - 1
                ))

                if scheduler:
                    scheduler.step(epoch)

                for mode in ['train', 'val']:
                    if mode == 'train':
                        network.train()  # Set model to training mode

                        loader = tqdm(train_loader, total=len(train_loader))

                        for batch_idx, (data, target) in enumerate(loader):

                            data = Variable(data.float())
                            target = Variable(target)

                            if torch.cuda.is_available():
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
                            for data, target in val_loader:

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

                        test_loss /= len(val_loader.dataset)
                        print('\n\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                            test_loss, correct, len(val_loader.dataset),
                            100. * correct / len(val_loader.dataset)))
        except KeyboardInterrupt:
            print("\n\n**********Training stopped prematurely.**********\n\n")
        finally:
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    #THIS IS FOR INFERENCE
    def predict(self):
        pass

    #TODO: this is for the test data
    def evaluate(self):
        pass
