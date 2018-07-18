__author__ = 'Caitrin'

"A criterion is encapsulated by a loss function (here I am borrowing terminology commonly used by pytorch users"


#Caitrin's running notes for planning
"""
What does a criterion do? Well, it describes the performance
"""
import abc
import torch.nn as nn


class Criterion(nn.Module):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        super(Criterion, self).__init__() #we want to import the nn.Module __init___

    @abc.abstractmethod
    def forward(self):
        pass

        #you're returning something that tells you how to move forward in the future??
