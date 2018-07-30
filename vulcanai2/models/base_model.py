__author__ = 'Caitrin'
"""
This file will describe the required components of any model, much as was previously done in the __init__ of net.py
This closely follows pytorch's torch.nn.module
Any method can also be extended by calling super, as will very likely be the case with forward - there are some procedures that are necessary for every model
"""

import abc
import torch.nn as nn

class Model(nn.module):

    @abc.abstractmethod
    def __init__(self): #TODO: this will be a lot of params.
        pass

    #THE BELOW WAS LARGELY TAKEN FROM https://github.com/Cadene/bootstrap.pytorch/blob/master/bootstrap/models/model.py purely for display purposes
    #TODO: so, obviously, this needs to be changed

    def eval(self):
        super(Model, self).train(mode=False)
        self.mode = 'eval'

    def train(self):
        super(Model, self).train(mode=True)
        self.mode = 'train'

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

    def prepare_batch(self, batch):
        if self.is_cuda:
            batch = self.cuda_tf()(batch)
        if self.mode == 'eval':
            batch = self.detach_tf()(batch)
        return batch

    @abc.abstractmethod
    def forward(self, batch):
        batch = self.prepare_batch(batch)
        net_out = self.network(batch)

        cri_out = {}
        if self.mode in self.criterions:
            cri_tmp = self.criterions[self.mode](net_out, batch)
            if cri_tmp is not None:
                cri_out = cri_tmp

        met_out = {}
        if self.mode in self.metrics:
            met_tmp = self.metrics[self.mode](cri_out, net_out, batch)
            if met_tmp is not None:
                met_out = met_tmp

        out = {}
        if type(net_out) is dict:
            for key, value in net_out.items():
                out[key] = value
        if type(cri_out) is dict:
            for key, value in cri_out.items():
                out[key] = value
        if type(met_out) is dict:
            for key, value in met_out.items():
                out[key] = value
        return out

    @abc.abstractmethod
    def state_dict(self, *args, **kwgs):
        state = {}
        state['network'] = self.network.state_dict(*args, **kwgs)
        state['criterions'] = {}
        for mode, criterion in self.criterions.items():
            if hasattr(criterion, '__parameters'):
                state['criterions'][mode] = criterion.state_dict(*args, **kwgs)
        state['metrics'] = {}
        for mode, metric in self.metrics.items():
            if hasattr(metric, '__parameters'):
                state['metrics'][mode] = metric.state_dict(*args, **kwgs)
        return state

    @abc.abstractmethod
    def load_state_dict(self, state, *args, **kwgs):
        self.network.load_state_dict(state['network'], *args, **kwgs)
        for mode, criterion in self.criterions.items():
            if hasattr(criterion, '__parameters'):
                criterion.load_state_dict(state['criterions'][mode], *args, **kwgs)
        for mode, metric in self.metrics.items():
            if hasattr(metric, '__parameters'):
                metric.load_state_dict(state['metrics'][mode], *args, **kwgs)
