# -*- coding: utf-8 -*-
""" Imports dataset classes so they can be used directly"""
from .fashion import FashionData
from .tabulardataset import TabularDataset
__all__ = [
    'fashion',
    'tabulardataset',
    'utils',
    'FashionData',
    'TabularDataset'
]
