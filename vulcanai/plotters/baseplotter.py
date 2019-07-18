__author__ = 'Caitrin'
"""

This file will describe the required components of any class used to provide plotting functionality
It is quite possible that we do not want this to be an abstract base class, as we do not require any functionality..
"""

import abc
import logging
logger = logging.getLogger(__name__)


class Plotter(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def display_something(self):
        """
        this is just a test.
        :return:
        """
