"""Contains auxilliary methods."""
import os

import numpy as np

import pandas as pd

from math import sqrt, ceil, floor

import pickle

from datetime import datetime

from .utils import GuidedBackprop
from ..models.utils import get_notable_indices

import torch

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib
import matplotlib.pyplot as plt


import itertools
import logging
logger = logging.getLogger(__name__)


def display_record(record=None, load_path=None, interactive=True):
    """
    Display the training curve for a network training session.

    :param record: the record dictionary for dynamic graphs during training
    :param load_path: the saved record .pickle file to load
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
    #val_error = [i if ~np.isnan(i) else None for i in record['validation_error']]
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

    if interactive is True:
        plt.draw()
        plt.pause(1e-17)

def display_pca(train_x, train_y, label_map=None):
    pca = PCA(n_components=2, random_state=0)
    x_transform = pca.fit_transform(train_x)
    _plot_reduction(
        x_transform,
        train_y,
        label_map=label_map,
        title='PCA Visualization')

def display_tsne(train_x, train_y, label_map=None):
    """
    t-distributed Stochastic Neighbor Embedding (t-SNE) visualization [1].

    [1]: Maaten, L., Hinton, G. (2008). Visualizing Data using t-SNE.
            JMLR 9(Nov):2579--2605.

    Args:
        train_x: 2d numpy array (batch, features) of samples
        train_y: 2d numpy array (batch, labels) for samples
        label_map: a dict of labelled (str(int), string) key, value pairs
    """
    tsne = TSNE(n_components=2, random_state=0)
    x_transform = tsne.fit_transform(train_x)
    _plot_reduction(
        x_transform,
        train_y,
        label_map=label_map,
        title='t-SNE Visualization')

def _plot_reduction(x_transform, train_y, label_map, title='Dim Reduction'):
    y_unique = np.unique(train_y)
    if label_map is None:
        label_map = {str(i): str(i) for i in y_unique}
    elif not isinstance(label_map, dict):
        raise ValueError('label_map most be a dict of a key'
                         ' mapping to its true label')
    colours = np.array(sns.color_palette("hls", len(y_unique)))
    plt.figure()
    for index, cl in enumerate(y_unique):
        plt.scatter(x=x_transform[train_y == cl, 0],
                    y=x_transform[train_y == cl, 1],
                    s=100,
                    c=colours[index],
                    alpha=0.5,
                    marker='o',
                    edgecolors='none',
                    label=label_map[str(cl)])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc='upper right')
    plt.title(title)
    plt.show(False)

def display_confusion_matrix(cm, class_list):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    inspired from: https://github.com/zaidalyafeai/Machine-Learning/blob/master/Multi-input%20Network%20Pytorch.ipynb
    
    Args:
        cm: confustion_matrix obtained using vulcanai.Metrics.get_confusion_matrix
        class_list: List of actual class labels (e.g.: MNIST - [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    """
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_list))
    plt.xticks(tick_marks, class_list, rotation=45)
    plt.yticks(tick_marks, class_list)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show(False)


def compute_saliency_map(network, input_x, input_y):
    """
    Return the saliency map using the guided backpropagation method [1].

    [1]: Springgenberg, J.T., Dosovitskiy, A., Brox, T., Riedmiller, M. (2015).
         Striving for Simplicity: The All Convolutional Net. ICLR 2015 
         (https://arxiv.org/pdf/1412.6806.pdf)

    :param network: A network type of subclass BaseNetwork
    :param input_x: Input array of shape (batch, channel, width, height) or 
                    (batch, channel, width, height, depth)
    :param input_y: 1D array with class targets
    :return: Top layer gradients of same shape as input data
    """
    guided_backprop = GuidedBackprop(network)
    saliency_map = guided_backprop.generate_gradients(input_x, input_y)
    guided_backprop.remove_hooks()
    # saliency_map, _ = torch.max(saliency_map, dim = 1) # get max abs from all channels
    return saliency_map


def display_saliency_overlay(image, saliency_map, shape=(28, 28)):
    """
    Plot overlay saliency map over image.

    :param image: numpy or torch array (1d 2d, or 3d) for single image
    :param saliency_map: numpy array (1d 2d, or 3d) for single image
    :param shape: the dimensions of the image. defaults to mnist.
    :return: None
    """
    # Handle different colour channels and shapes for image input
    if len(image.shape) == 3:
        if image.shape[0] == 1:
            # For 1 colour channel, remove it
            image = image[0]
        elif image.shape[0] == 3 or image.shape[0] == 4:
            # For 3 or 4 colour channels, move to end for plotting
            image = np.moveaxis(image, 0, -1)
        else:
            raise ValueError("Invalid number of colour channels in input.")
    elif len(image.shape) == 1:
        image = np.reshape(image, shape)

    # Handle different colour channels and shapes for saliency map
    if len(saliency_map.shape) == 3:
        if saliency_map.shape[0] == 1:
            # For 1 colour channel, remove it
            saliency_map = saliency_map[0]
        elif saliency_map.shape[0] == 3 or saliency_map.shape[0] == 4:
            # For 3 or 4 colour channels, move to end for plotting
            saliency_map = np.moveaxis(saliency_map, 0, -1)
        else:
            raise ValueError("Invalid number of colour channels in saliency map.")
    elif len(saliency_map.shape) == 1:
        saliency_map = np.reshape(saliency_map, shape)

    fig = plt.figure()
    # Plot original image
    fig.add_subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    # Plot original with saliency overlay
    fig.add_subplot(1, 2, 2)
    plt.imshow(image, cmap='binary')
    # Optional: get the absolute values of the saliency map here
    plt.imshow(saliency_map, cmap='Blues', alpha=0.7)
    plt.colorbar()
    plt.show(False)


def display_receptive_fields(network, top_k=5):
    """
    Display receptive fields of layers from a network [1].

    [1]: Luo, W., Li, Y., Urtason, R., Zemel, R. (2016).
         Understanding the Effective Receptive Field in Deep
         Convolutional Neural Networks. Advances in Neural Information
         Processing Systems, 29 (NIPS 2016)


    :param network: Network object
    :param top_k: most and least k important features from field

    Returns a dict of the top k and bottom k important features.
    """
    if type(network).__name__ == "ConvNet":
        raise NotImplementedError("ConvNet receptive fields not yet implemented")
    elif '_input_network' in network._modules:
        if type(network._modules['_input_network']).__name__ == "ConvNet":
            raise NotImplementedError("ConvNet receptive fields not yet implemented")

    feature_importance = {}
    fig = plt.figure()
    fig.suptitle("Feature importance")
    num_layers = len(network._modules['network'])
    for i, layer in enumerate(network._modules['network']):
        raw_field = layer.kernel._parameters['weight'].detach()
        field = np.average(raw_field, axis=0)  # average all outgoing
        field_shape = [
            floor(sqrt(field.shape[0])),
            ceil(sqrt(field.shape[0]))
            ]
        fig.add_subplot(
            floor(sqrt(num_layers)),
            ceil(sqrt(num_layers)),
            i + 1)
        feats = get_notable_indices(abs(field), top_k=top_k)
        unit_type = type(layer).__name__
        layer_name = '{}_{}'.format(unit_type, i)
        feature_importance.update({layer_name: feats})
        plt.title(layer_name)
        plt.imshow(np.resize(abs(field), field_shape), cmap='hot_r')
        plt.colorbar()
    plt.show(False)
    return feature_importance