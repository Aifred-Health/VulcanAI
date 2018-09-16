#TODO: all methods need to be updated to work with new dataset
__author__="RobertFratila"

"""Contains auxilliary methods."""
import os

import numpy as np

import pandas as pd

from math import sqrt, ceil, floor

import pickle

from datetime import datetime

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)


def display_record(record=None, load_path=None):
    """
    Display the training curve for a network training session.

    Args:
        record: the record dictionary for dynamic graphs during training
        load_path: the saved record .pickle file to load
    """
    title = 'Training curve'
    if load_path is not None:
        with open(load_path) as in_file:
            record = pickle.load(in_file)
        title = 'Training curve for model: {}'.format(
            os.path.basename(load_path))

    if record is None or not isinstance(record, dict):
        raise ValueError('No record exists and cannot be displayed.')

    plt.subplot(1, 2, 1)
    plt.title("{}: Error".format(title))
    train_error, = plt.plot(
        record['epoch'],
        record['train_error'],
        '-mo',
        label='Train Error'
    )
    validation_error, = plt.plot(
        record['epoch'],
        record['validation_error'],
        '-ro',
        label='Validation Error'
    )
    plt.xlabel("Epoch")
    plt.ylabel("Cross entropy error")
    plt.legend(handles=[train_error,
                        validation_error],
               loc=0)

    plt.subplot(1, 2, 2)
    plt.title("{}: Accuracy".format(title))
    train_accuracy, = plt.plot(
        record['epoch'],
        record['train_accuracy'],
        '-go',
        label='Train Accuracy'
    )
    validation_accuracy, = plt.plot(
        record['epoch'],
        record['validation_accuracy'],
        '-bo',
        label='Validation Accuracy'
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    plt.legend(handles=[train_accuracy,
                        validation_accuracy],
               loc=0)

    plt.show(False)
    plt.pause(0.0001)



