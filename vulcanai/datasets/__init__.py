# -*- coding: utf-8 -*-
""" Imports dataset classes so they can be used directly"""
import .fashion
from .fashion import FashionData
import .tabulardataset
from .tabulardataset import TabularDataset
import .multidataset
from .multidataset import MultiDataset

__all__ = [
    'fashion',
    'tabulardataset',
    'utils',
    'FashionData',
    'TabularDataset',
    'MultiDataset'
]
