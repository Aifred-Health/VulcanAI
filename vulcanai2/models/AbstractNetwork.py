__author__ = 'Caitrin'

import abc
import torch.nn as nn
import datetime
import torch.nn.modules.activation as activations #TODO: want to call it activation but namespace, so what to do best?
from torch import optim
import torch.nn.modules.loss as loss

class AbstractNetwork(nn.module):

    #TODO: this should be the same for every model, given that you pass a config??
    #TODO: come up with an alternate way to validate config
    #TODO: deal with stopping rules
    #TODO: do you have to call any nn.module methods?? do you even actually want to subclass at this point if you have layers that subclass nn.module??
    def __init__(self, name, dimensions, config, save_path=None, input_network=None, num_classes=None, activation=activations.Softmax, pred_activation=activations.Softmax, optimizer=optim.Adam, learning_rate=0.001, lr_scheduler=None, stopping_rule='best_validation_error', criterion=None):
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
        super(AbstractNetwork, self).__init__()
        self.name = name
        self.dimensions = dimensions
        self.config = config

        if not save_path:
            self.save_path = "{}_{date:%Y-%m-%d_%H:%M:%S}/".format(self.name, date=datetime.datetime.now())
        else:
            self.save_path = save_path

        self.input_network = input_network #TODO: change and check type here?
        self.num_classes = num_classes
        self.activation = activation
        self.pred_activation = pred_activation

        #TODO: how to deal with passing parameters especially if custom, given the note here: https://pytorch.org/docs/stable/optim.html
        #TODO: some optmizers have different behaviour, although this is unlikely to apply
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.stopping_rule = stopping_rule

        if not criterion and not self.num_classes:
            self.criterion = loss.MSELoss
        elif not criterion:
            self.criterion = loss.CrossEntropyLoss


    #TODO: figure out how this works in conjunction with optimizer
    #TODO: fix the fact that you copy pasted this
    def cuda(self, device_id=None):
        """Moves all model parameters and buffers to the GPU.
        Arguments:
            device_id (int, optional): if specified, all parameters will be
                copied to that device
        """
        self.is_cuda = True
        return self._apply(lambda t: t.cuda(device_id))

    def cpu(self):
        """Moves all model parameters and buffers to the CPU."""
        self.is_cuda = False
        return self._apply(lambda t: t.cpu())

    @abc.abstractmethod
    def forward(self, batch): #TODO: uhhhh do you want to define this if it's really self? do you want to subclass nn.module??
        pass

    #TODO: deal with the fact that you copied this
    def prepare_batch(self, batch):
        if self.is_cuda:
            batch = self.cuda_tf()(batch)
        if self.mode == 'eval':
            batch = self.detach_tf()(batch)
        return batch