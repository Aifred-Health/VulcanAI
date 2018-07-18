__author__ = 'Caitrin'


"""
normally just use the pytorch ones... which are returned appropriately in factory.
you could store your options in a list somewhere just like the bootstrap package?

perhaps call the directory custom_optimizers?
"""


"""
Not sure yet if we want this to return torch.optim or a custom wrapper class


How do you want it so that you can just use the implemented ones... well I guess you could have it return an object of the same type
and normally you just use their methods, but when you want something custom you follow the interfact.
then the factory deals with returning what you want.

woo inheritance
"""

from torch.optim.optimizer import Optimizer
import abc

class CustomOptimizer(Optimizer):


    @abc.abstractmethod #because there will be many more params
    def __init__(self, params, defaults):
        super(CustomOptimizer, self).__init__(params, defaults)

    @abc.abstractmethod
    def step(self, closure=None):
        loss = None
        return loss