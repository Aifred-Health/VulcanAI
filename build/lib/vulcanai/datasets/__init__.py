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
from .multidataset import MultiDataset

__all__ = [
    'fashion',
    'tabular_data_utils.py',
    'utils',
    'FashionData',
    'MultiDataset'
]