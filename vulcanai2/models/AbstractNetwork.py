__author__ = 'Caitrin'

import abc
import torch.nn as nn
import datetime
import torch.nn.modules.activation as activations #TODO: want to call it activation but namespace, so what to do best?
from torch import optim
import torch.nn.modules.loss as loss
from .Layers import * #TODO: blarg

class AbstractNetwork():

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

        self.network = nn.Sequential()
        self.create_network()


    #TODO: define setters for everything and call those... although I really don't care about making things private?


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

    #TODO: I think this needs to take into account passing through networks
    @abc.abstractmethod
    def create_network(self):
        pass

    #TODO: deal with the fact that you copied this
    #TODO: figure out if you really need this?
    def prepare_batch(self, batch):
        if self.is_cuda:
            batch = self.cuda_tf()(batch)
        if self.mode == 'eval':
            batch = self.detach_tf()(batch)
        return batch

    def get_all_layers(self):
        layers = []
        for l_name, l in self.input_network['network'].network.named_children():
            if isinstance(l, nn.Sequential):
                for subl_name, subl in l.named_children():
                    layers.append(subl)
            else:
                for param in l.parameters():
                    self.input_dimensions= param.size(0)



    def create_classification_layer(self, num_classes,
                                    nonlinearity):
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
        layer = DenseUnit(
                          in_channels=self.input_dim,
                          out_channels=num_classes,
                          bias=True,
                          activation=nonlinearity,
                          norm=None)
        self.network.add_module(layer_name, layer)
        self.layers.append(layer)
        print('\t\t{}'.format(layer))
