__author__ = 'Caitrin'
"""

The engine deals with training, evaluating, saving and checkpointing. Most of the code will be written in this class, but can be overridden where necessary

"""

import abc

class Engine(object):

    __metaclass__ = abc.ABCMeta


    @abc.abstractmethod
    def __init__(self): #TODO: this will be a lot of params.
        pass

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def train_epoch(self):
        pass

    @abc.abstractmethod
    def eval_epoch(self):
        pass

    @abc.abstractmethod
    def load_checkpoint(self):
        pass

    @abc.abstractmethod
    def save_checkpoint(self):
        pass
