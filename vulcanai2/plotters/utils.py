#FROM ORIGINAL VULCAN
#FROM ORIGINAL VULCAN
author = Robert Fratila

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

if os.name is not "posix":
    if "DISPLAY" not in os.environ:
        matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



#TODO: update to work with pytorch
# def display_pca(train_x, train_y, label_map=None):
#     pca = PCA(n_components=2, random_state=0)
#     x_transform = pca.fit_transform(train_x)
#     _plot_reduction(
#         x_transform,
#         train_y,
#         label_map=label_map,
#         title='PCA Visualization')
#
#
# def display_tsne(train_x, train_y, label_map=None):
#     """
#     t-distributed Stochastic Neighbor Embedding (t-SNE) visualization [1].
#
#     [1]: Maaten, L., Hinton, G. (2008). Visualizing Data using t-SNE.
#             JMLR 9(Nov):2579--2605.
#
#     Args:
#         train_x: 2d numpy array (batch, features) of samples
#         train_y: 2d numpy array (batch, labels) for samples
#         label_map: a dict of labelled (str(int), string) key, value pairs
#     """
#     tsne = TSNE(n_components=2, random_state=0)
#     x_transform = tsne.fit_transform(train_x)
#     _plot_reduction(
#         x_transform,
#         train_y,
#         label_map=label_map,
#         title='t-SNE Visualization')
#
#
# def _plot_reduction(x_transform, train_y, label_map, title='Dim Reduction'):
#     y_unique = np.unique(train_y)
#     if label_map is None:
#         label_map = {str(i): str(i) for i in y_unique}
#     elif not isinstance(label_map, dict):
#         raise ValueError('label_map most be a dict of a key'
#                          ' mapping to its true label')
#     colours = plt.cm.rainbow(np.linspace(0, 1, len(y_unique)))
#     plt.figure()
#     for index, cl in enumerate(y_unique):
#         plt.scatter(x=x_transform[train_y == cl, 0],
#                     y=x_transform[train_y == cl, 1],
#                     s=100,
#                     c=colours[index],
#                     alpha=0.5,
#                     marker='o',
#                     edgecolors='none',
#                     label=label_map[str(cl)])
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.legend(loc='upper right')
#     plt.title(title)
#     plt.show(False)
#
#
# def display_receptive_fields(network, layer_list=None, top_k=5):
#     """
#     Display receptive fields of layers from a network [1].
#
#     [1]: Luo, W., Li, Y., Urtason, R., Zemel, R. (2016).
#          Understanding the Effective Receptive Field in Deep
#          Convolutional Neural Networks. Advances in Neural Information
#          Processing Systems, 29 (NIPS 2016)
#
#
#     Args:
#         network: Network object
#         layer: list of layer indices to get fields from
#         top_k: most and least k important features from field
#
#     Returns a dict of the top k and bottom k important features.
#     """
#     if layer_list is None:
#         layer_list = range(len(network.layers))
#     if not isinstance(layer_list, list):
#         raise ValueError('layer_list must be a list of int')
#     layers = []
#     # Filter out layers with no weights. e.g. input, dropout
#     for index in layer_list:
#         try:
#             network.layers[index].W
#         except:
#             print ('Skipping layer: {}'.format(network.layers[index].name))
#             continue
#         else:
#             layers.append(network.layers[index])
#     # Get fields for filtered layers
#     feature_importance = {}
#     fig = plt.figure()
#     for i, l in enumerate(layers):
#         raw_field = l.W.container.storage[0]
#         field = np.average(raw_field, axis=1)  # average all outgoing
#         field_shape = [int(sqrt(field.shape[0]))] * 2
#         fig.add_subplot(floor(sqrt(len(layers))),
#                         ceil(sqrt(len(layers))),
#                         i + 1)
#         feats = get_notable_indices(abs(field), top_k=top_k)
#         feature_importance.update({'{}'.format(l.name): feats})
#         plt.title(l.name)
#         plt.imshow(np.reshape(abs(field), field_shape), cmap='hot_r')
#         plt.colorbar()
#     plt.show(False)
#     return feature_importance
#
#
# def get_notable_indices(matrix, top_k=5):
#     """
#     Return dict of top k and bottom k features useful from matrix.
#
#     Args:
#         matrix: 1d numpy array
#         top_k: defaults to top and bottom 5 indices
#     """
#     important_features = matrix.argsort()[-top_k:][::-1]
#     unimportant_features = matrix.argsort()[:-1][:top_k]
#     return {'important_indices': important_features,
#             'unimportant_indices': unimportant_features}
#
#
# def display_saliency_overlay(image, saliency_map, shape=(28, 28)):
#     """
#     Plot overlay saliency map over image.
#
#     Args:
#         image: numpy array (1d vector) for single image
#         saliency_map: numpy array (1d vector) for image
#         shape: the dimensions of the image. defaults to mnist.
#     """
#     if len(image.shape) == 3 or len(saliency_map.shape) == 3:
#         image = image[0]
#         saliency_map = saliency_map[0]
#     elif len(image.shape) == 1 or len(saliency_map.shape) == 1:
#         image = np.reshape(image, shape)
#         saliency_map = np.reshape(saliency_map, shape)
#
#     fig = plt.figure()
#     fig.add_subplot(1, 2, 1)
#     plt.imshow(image, cmap='gray')
#     fig.add_subplot(1, 2, 2)
#     plt.imshow(image, cmap='binary')
#     plt.imshow(abs(saliency_map), cmap='hot_r', alpha=0.7)
#     plt.colorbar()
#     plt.show(False)
#
#
# def get_saliency_map(network, input_data):
#     """
#     Calculate the saliency map for all input samples.
#
#     Calculates the derivative of the score w.r.t the input.
#     Helps with getting the 'why' from a prediction.
#
#     Args:
#         network: Network type to get saliency from
#         input_data: ndarray(2D), batch of samples
#
#     Returns saliency map for all given samples
#     """
#     output = lasagne.layers.get_output(
#         network.layers[-2],
#         deterministic=True
#     )
#     max_out = T.max(output, axis=1)
#     sal_fun = theano.function(
#         [network.input_var],
#         T.grad(max_out.sum(), wrt=network.input_var)
#     )
#     sal_map = sal_fun(input_data)
#     if sal_map.shape != input_data.shape:
#         raise ValueError('Shape mismatch')
#     return sal_map
#
#
# def get_all_embedded_networks(network):
#     """
#     Return all embedded networks of type Network.
#
#     Args:
#         network: tallest point, hierarchically, of which to begin
#             gathering the embedded networks
#
#     Returns: a list of Networks in order of their stack
#         example: if we have a model a->b->c, it will return
#         [c,b,a]. the specific layer that was attached can be extracted
#         from the individual network itself.
#     """
#     if network.input_network is None:
#         return [network]
#     else:
#         return [network] + \
#                get_all_embedded_networks(network.input_network['network'])
#
#
# def round_list(raw_list, decimals=4):
#     """
#     Return the same list with each item rounded off.
#
#     Args:
#         raw_list: float list
#         decimals: how many decimal points to round to
#
#     Returns: the rounded list
#     """
#     return [round(item, decimals) for item in raw_list]
#
#
# def get_confusion_matrix(prediction, truth):
#     """
#     Calculate the confusion matrix for classification network predictions.
#
#     Args:
#         predicted: the class matrix predicted by the network.
#                    Does not take one hot vectors.
#         actual: the class matrix of the ground truth
#                 Does not take one hot vectors.
#
#     Returns: the confusion matrix
#     """
#     if len(prediction.shape) == 2:
#         prediction = prediction[:, 0]
#     if len(truth.shape) == 2:
#         truth = truth[:, 0]
#
#     return confusion_matrix(y_true=truth,
#                             y_pred=prediction)
#
#
# def get_one_hot(in_matrix):
#     """
#     Reformat truth matrix to same size as the output of the dense network.
#
#     Args:
#         in_matrix: the categorized 1D matrix
#
#     Returns: a one-hot matrix representing the categorized matrix
#     """
#     if in_matrix.dtype.name == 'category':
#         custum_array = in_matrix.cat.codes
#
#     elif isinstance(in_matrix, np.ndarray):
#         custum_array = in_matrix
#
#     else:
#         raise ValueError("Input matrix cannot be converted.")
#
#     lb = LabelBinarizer()
#     return np.array(lb.fit_transform(custum_array), dtype='float32')
#
#
# def get_class(in_matrix):
#     """
#     Reformat truth matrix to be the classes in a 1D array.
#
#     Args:
#         in_matrix: one-hot matrix
#
#     Returns: Class array
#     """
#     if in_matrix.shape[1] > 1:
#         return np.expand_dims(np.argmax(in_matrix, axis=1), axis=1)
#     elif in_matrix.shape[1] == 1:
#         return np.around(in_matrix)
#
#
# def display_record(record=None, load_path=None):
#     """
#     Display the training curve for a network training session.
#
#     Args:
#         record: the record dictionary for dynamic graphs during training
#         load_path: the saved record .pickle file to load
#     """
#     title = 'Training curve'
#     if load_path is not None:
#         with open(load_path) as in_file:
#             record = pickle.load(in_file)
#         title = 'Training curve for model: {}'.format(
#             os.path.basename(load_path))
#
#     if record is None or not isinstance(record, dict):
#         raise ValueError('No record exists and cannot be displayed.')
#
#     plt.subplot(1, 2, 1)
#     plt.title("{}: Error".format(title))
#     train_error, = plt.plot(
#         record['epoch'],
#         record['train_error'],
#         '-mo',
#         label='Train Error'
#     )
#     validation_error, = plt.plot(
#         record['epoch'],
#         record['validation_error'],
#         '-ro',
#         label='Validation Error'
#     )
#     plt.xlabel("Epoch")
#     plt.ylabel("Cross entropy error")
#     plt.legend(handles=[train_error,
#                         validation_error],
#                loc=0)
#
#     plt.subplot(1, 2, 2)
#     plt.title("{}: Accuracy".format(title))
#     train_accuracy, = plt.plot(
#         record['epoch'],
#         record['train_accuracy'],
#         '-go',
#         label='Train Accuracy'
#     )
#     validation_accuracy, = plt.plot(
#         record['epoch'],
#         record['validation_accuracy'],
#         '-bo',
#         label='Validation Accuracy'
#     )
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     plt.ylim(0, 1)
#
#     plt.legend(handles=[train_accuracy,
#                         validation_accuracy],
#                loc=0)
#
#     plt.show(False)
#     plt.pause(0.0001)
#
#
# def get_timestamp():
#     """Return a 14 digit timestamp."""
#     return datetime.now().strftime('%Y%m%d%H%M%S_')


