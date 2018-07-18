__author__ = 'Caitrin'
"""

This class allows us format our data consistently no matter the source
"""

import abc

class DataLoader(object): #TODO: I think this should subclass the torch dataset??

    __metaclass__ = abc.ABCMeta


    @abc.abstractmethod
    def __init__(self): #TODO: this will be a lot of params.
        pass

