"""Contains all visualization methods."""
import os

import numpy as np

from math import sqrt, ceil, floor
import pickle
from datetime import datetime
from .utils import GuidedBackprop, get_notable_indices

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import seaborn as sns

import itertools
import logging
logger = logging.getLogger(__name__)

DISPLAY_AVAILABLE = True if os.environ.get("DISPLAY") else False


def save_visualization(plot, path=None):
    """
    Save plot at designated path.

    Parameters:
        plot : matplotlib
            Matplotlib variable with savefig ability.
        path : string
            String that designates the path to save the given figure to.

    Returns:
        None

    """
    plot.savefig(path)
    logger.info("Saved visualization at %s", path)


def get_save_path(path, vis_type):
    """Return a save_path string."""
    path = "{}{}_{date:%Y-%m-%d_%H-%M-%S}.png".format(
        path, vis_type, date=datetime.now())
    return path


def display_record(record=None, save_path=None, interactive=True):
    """
    Display the training curve for a network training session.

    Parameters:
        record : dict
            the network record dictionary for dynamic graphs during training.
        save_path : String
            String that designates the path to save figure to be produced.
            Save_path must be a proper path that ends with a filename with an
            image filetype.
        interactive : boolean
            To display during training or afterwards.

    Returns:
        None

    """
    title = 'Training curve'
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
    # val_error = \
    # [i if ~np.isnan(i) else None for i in record['validation_error']]
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

    if save_path:
        save_visualization(plt, save_path)

    if not DISPLAY_AVAILABLE and save_path is None:
        raise RuntimeError(
            "No display environment found. "
            "Display environment needed to plot, "
            "or set save_path=path/to/dir")

    elif interactive is True:
        plt.draw()
        plt.pause(1e-17)


def display_pca(input_data, targets, label_map=None, save_path=None):
    """
    Calculate pca reduction and plot it.

    Parameters:
        input_data : numpy.dnarray
            Input data to reduce in dimensions.
        targets : numpy.ndarray
            size (batch, labels) for samples.
        label_map : dict
            labelled {str(int), string} key, value pairs.
        save_path : String
            String that designates the path to save figure to be produced.

    """
    pca = PCA(n_components=2, random_state=0)
    x_transform = pca.fit_transform(input_data)
    _plot_reduction(
        x_transform,
        targets,
        label_map=label_map,
        title='PCA Visualization',
        save_path=save_path)


def display_tsne(input_data, targets, label_map=None, save_path=None):
    """
    t-distributed Stochastic Neighbor Embedding (t-SNE) visualization [1].

    [1]: Maaten, L., Hinton, G. (2008). Visualizing Data using t-SNE.
            JMLR 9(Nov):2579--2605.

    Parameters:
        input_data : numpy.dnarray
            Input data to reduce in dimensions.
        targets : numpy.ndarray
            size (batch, labels) for samples.
        label_map : dict
            labelled {str(int), string} key, value pairs.
        save_path : String
            String that designates the path to save figure to be produced.

    """
    tsne = TSNE(n_components=2, random_state=0)
    x_transform = tsne.fit_transform(input_data)
    _plot_reduction(
        x_transform,
        targets,
        label_map=label_map,
        title='t-SNE Visualization',
        save_path=save_path)


def _plot_reduction(x_transform, targets, label_map, title, save_path=None):
    """Once PCA and t-SNE has been calculated, this is used to plot."""
    y_unique = np.unique(targets)
    if label_map is None:
        label_map = {str(i): str(i) for i in y_unique}
    elif not isinstance(label_map, dict):
        raise ValueError('label_map most be a dict of a key'
                         ' mapping to its true label')
    colours = np.array(sns.color_palette("hls", len(y_unique)))
    plt.figure()
    for index, cl in enumerate(y_unique):
        plt.scatter(x=x_transform[targets == cl, 0],
                    y=x_transform[targets == cl, 1],
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

    if save_path:
        save_path = get_save_path(save_path, vis_type=title)
        save_visualization(plt, save_path)

    if not DISPLAY_AVAILABLE and save_path is None:
        raise RuntimeError(
            "No display environment found. "
            "Display environment needed to plot, "
            "or set save_path=path/to/dir")
    else:
        plt.show(False)


def display_confusion_matrix(cm, class_list=None, save_path=None):
    """
    Print and plot the confusion matrix.

    inspired from: https://github.com/zaidalyafeai/Machine-Learning

    Parameters:
        cm : numpy.ndarray
            2D confustion_matrix obtained using utils.get_confusion_matrix
        class_list : list
            Actual class labels (e.g.: MNIST - [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        save_path : String
            String that designates the path to save figure to be produced.

    """
    if class_list is None:
        class_list = list(range(cm.shape[0]))
    if not isinstance(class_list, list):
        raise ValueError("class_list must be of type list.")
    plt.figure()
    ax = plt.gca()
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues', origin='lower')

    plt.title('Confusion matrix')
    tick_marks = np.arange(len(class_list))
    plt.xticks(tick_marks, class_list, rotation=45)
    plt.yticks(tick_marks, class_list)
    # Plot number overlay
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    # Plot labels
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.03)
    plt.colorbar(im, cax=cax)

    if save_path:
        save_path = get_save_path(save_path, vis_type='confusion_matrix')
        save_visualization(plt, save_path)

    if not DISPLAY_AVAILABLE and save_path is None:
        raise RuntimeError(
            "No display environment found. "
            "Display environment needed to plot, "
            "or set save_path=path/to/dir")
    else:
        plt.show(False)


def compute_saliency_map(network, input_data, targets):
    """
    Return the saliency map using the guided backpropagation method [1].

    [1]: Springgenberg, J.T., Dosovitskiy, A., Brox, T., Riedmiller, M. (2015).
         Striving for Simplicity: The All Convolutional Net. ICLR 2015
         (https://arxiv.org/pdf/1412.6806.pdf)

    Parameters:
        network : BaseNetwork
            A network to get saliency maps on.
        input_data : numpy.ndarray
            Input array of shape (batch, channel, width, height) or
            (batch, channel, width, height, depth).
        targets : numpy.ndarray
            1D array with class targets of size [batch].

    Returns:
        saliency_map : list of numpy.ndarray
            Top layer gradients of the same shape as input data.

    """
    guided_backprop = GuidedBackprop(network)
    saliency_map = guided_backprop.generate_gradients(input_data, targets)
    return saliency_map


def display_saliency_overlay(image, saliency_map, shape=(28, 28), save_path=None):
    """
    Plot overlay saliency map over image.

    Parameters:
        image : numpy.ndarray
            (1D, 2D, 3D) for single image or linear output.
        saliency_map: numpy.ndarray
            (1D, 2D, 3D) for single image or linear output.
        shape : tuple, list
            The dimensions of the image. Defaults to mnist.
        save_path : String
            String that designates the path to save figure to be produced.

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
            raise ValueError("Invalid number of channels in saliency map.")
    elif len(saliency_map.shape) == 1:
        saliency_map = np.reshape(saliency_map, shape)

    fig = plt.figure()
    fig.suptitle("Saliency Map")
    # Plot original image
    fig.add_subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    # Plot original with saliency overlay
    fig.add_subplot(1, 2, 2)
    plt.imshow(image, cmap='binary')

    ax = plt.gca()
    im = ax.imshow(saliency_map, cmap='Blues', alpha=0.7)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.03)
    plt.colorbar(im, cax=cax, format='%.0e')

    if save_path:
        save_path = get_save_path(save_path, vis_type='saliency_map')
        save_visualization(plt, save_path)

    if not DISPLAY_AVAILABLE and save_path is None:
        raise RuntimeError(
            "No display environment found. "
            "Display environment needed to plot, "
            "or set save_path=path/to/dir")
    else:
        plt.show(False)


def display_receptive_fields(network, top_k=5, save_path=None):
    """
    Display receptive fields of layers from a network [1].

    [1]: Luo, W., Li, Y., Urtason, R., Zemel, R. (2016).
         Understanding the Effective Receptive Field in Deep
         Convolutional Neural Networks. Advances in Neural Information
         Processing Systems, 29 (NIPS 2016)

    Parameters:
        network : BaseNetwork
            Network to get receptive fields of.
        top_k : int
            To return the most and least k important features from field
        save_path : String
            String that designates the path to save figure to be produced.

    Returns:
        k_features: dict
            A dict of the top k and bottom k important features.

    """
    if type(network).__name__ == "ConvNet":
        raise NotImplementedError
    elif '_input_network' in network._modules:
        if type(network._modules['_input_network']).__name__ == "ConvNet":
            raise NotImplementedError

    feature_importance = {}
    fig = plt.figure()
    fig.suptitle("Feature importance")
    num_layers = len(network._modules['network'])
    for i, layer in enumerate(network._modules['network']):
        raw_field = layer._kernel._parameters['weight'].cpu().detach().numpy()
        field = np.average(raw_field, axis=0)  # average all outgoing
        field_shape = [
            floor(sqrt(field.shape[0])),
            ceil(sqrt(field.shape[0]))
            ]
        fig.add_subplot(
            floor(sqrt(num_layers)),
            ceil(sqrt(num_layers)),
            i + 1)
        field = abs(field)
        feats = get_notable_indices(field, top_k=top_k)
        unit_type = type(layer).__name__
        layer_name = '{}_{}'.format(unit_type, i)
        feature_importance.update({layer_name: feats})
        plt.title(layer_name)
        plt.imshow(np.resize(field, field_shape), cmap='Blues')
        plt.colorbar()

    if save_path:
        save_path = get_save_path(save_path, vis_type='feature_importance')
        save_visualization(plt, save_path)

    if not DISPLAY_AVAILABLE and save_path is None:
        raise RuntimeError(
            "No display environment found. "
            "Display environment needed to plot, "
            "or set save_path=path/to/dir")
    else:
        plt.show(False)

    return feature_importance
