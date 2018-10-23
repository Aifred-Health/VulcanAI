# coding=utf-8

import torch
import torch.nn as nn

from .basenetwork import BaseNetwork
from .cnn import ConvNet
from .dnn import DenseNet

import numpy as np
import logging

logger = logging.getLogger(__name__)


class MultiInputNN(ConvNet, DenseNet):

    def __init__(self, name, input_models):
        """
        Concatenates the input models to one model
        """
        
        nn.Module.__init__(self)
        super(MultiInputNN, self).__init__(name, input_models)

        self.input_models = input_models

    def forward(self, *x):

        assert len(self.input_models) == len(x), "Number of models and number of input does not match"

        outputs = []
        for i in range(len(x)):
            outputs[i] = self.models[i](x[i])

        network_output = torch.cat(outputs, 1)
        
        return network_output

    def __str__(self):
        if self.optim:
            return super(MultiInputNN, self).__str__() + f'\noptim: {self.optim}'
        else:
            return super(MultiInputNN, self).__str__()
