# -*- coding: utf-8 -*-
""" Imports dataset classes so they can be used directly

Submodules
==========

.. autosummary::
    :toctree: _autosummary

    fashion
    tabulardataset
    multidataset
"""

from .fashion import FashionData
from .tabulardataset import TabularDataset
from .multidataset import MultiDataset

__all__ = [
    'fashion',
    'tabulardataset',
    'utils',
    'FashionData',
    'TabularDataset',
    'MultiDataset'
]

def start():
    " Start module "
