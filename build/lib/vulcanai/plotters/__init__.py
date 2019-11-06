# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from .visualization import (
    compute_saliency_map,
    display_saliency_overlay,
    display_pca,
    display_tsne,
    display_confusion_matrix,
    display_record,
    display_receptive_fields
)

__all__ = [
    'utils',
    'visualization',
    'compute_saliency_map',
    'display_saliency_overlay',
    'display_pca',
    'display_tsne',
    'display_confusion_matrix',
    'display_record',
    'display_receptive_fields',
    ]
