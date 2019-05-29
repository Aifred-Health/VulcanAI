# coding=utf-8
"""Defines the DenseNet class."""
import torch
import torch.nn as nn
from torch.autograd import Variable as V

from .basenetwork import BaseNetwork
from .layers import DenseUnit, FlattenUnit

import logging
from inspect import getfullargspec

from collections import OrderedDict

logger = logging.getLogger(__name__)

class LSTMNet(BaseNetwork):
    """Defines the necessary configuration for a LSTM Network."""

    def __init__(self, name, config, in_dim=None, save_path=None,
                 input_networks=None, num_classes=None,
                 activation=nn.ReLU(), pred_activation=None,
                 optim_spec={'name': 'Adam', 'lr': 0.001},
                 lr_scheduler=None, early_stopping=None,
                 criter_spec=nn.CrossEntropyLoss(),
                 device="cuda:0"):
        """Define the LSTMNet object."""
        super(LSTMNet, self).__init__(
            name, config, in_dim, save_path, input_networks,
            num_classes, activation, pred_activation, optim_spec,
            lr_scheduler, early_stopping, criter_spec, device)
   
    def _create_network(self, **kwargs):
        lstm_layers = self._config['hid_units']
        self.network = nn.LSTM(self.in_dim[0], 
                               self._config['hid_units'][0], 
                               self._config['num_layers'],
                               batch_first = self._config['batch_first'],
                               dropout=self._config['dropout'][0])
        #h0 initial hidden state
        h0 = torch.zeros(self._config['num_layers'], 
                           self._config['batch_size'], 
                           self._config['hid_units'][0])

        #c0 initialize cell state
        c0 = torch.zeros(self._config['num_layers'],
                           self._config['batch_size'],
                           self._config['hid_units'][0])
        self.hidden = (h0, c0)
        self.rnn_out_layer = nn.Linear(self._config['hid_units'][0], self._config['out_dim'])

    def __str__(self):
        if self.optim is not None:
            return super(LSTMNet, self).__str__() + '\noptim: {}'.format(self.optim)
        else:
            return super(LSTMNet, self).__str__()
