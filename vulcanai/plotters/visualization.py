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

from graphviz import Digraph
import torch
from torch.autograd import Variable

from collections import defaultdict

DISPLAY_AVAILABLE = True if os.environ.get("DISPLAY") else False

THEMES = {
    "basic": {
        "background_color": "#FFFFFF",
        "fill_color": "#E8E8E8",
        "outline_color": "#000000",
        "font_color": "#000000",
        "font_name": "Times",
        "font_size": "10",
        "margin": "0,0",
        "padding":  "1.0,0.5",
    },
    "blue": {
        "background_color": "#FFFFFF",
        "fill_color": "#BCD6FC",
        "outline_color": "#7C96BC",
        "font_color": "#202020",
        "font_name": "Verdana",
        "font_size": "10",
        "margin": "0,0",
        "padding":  "1.0,0.5",
    },
}


class Graph(object):
    """ 
    Builds the graph (dot) for the BaseNetwork model and
    saves the graph source code (.gv) in the current directory.
    
    #NOTE: Inspired from: https://github.com/szagoruyko/pytorchviz/blob/master/torchviz/dot.py
    and https://github.com/waleedka/hiddenlayer
    #TODO: Modify the saving directory.
    
    Parameters:
        model : BaseNetwork
            The model for which graph is built.
        input : list of torch.Tensor or torch.Tensor
            The input data to the BaseNetwork. Must be a list if the model is
            a multi-input type BaseNetwork.
        theme_color : dict
            The color schema for the model's graph.
        node_shape : str
            Sets the shape of the graph nodes.

    Example:

        >>> graph = Graph(model, x, theme_color='blue', node_shape='oval')
        >>> graph.view() # Saves a .pdf of the graph.dot object and opens the saved file
        using the pdf viewer

    """

    def __init__(self, model, input, theme_color='basic',
                 node_shape='ellipse'):
        self.model = model
        self.input = input
        self.params = dict(model.named_parameters())

        self.layer_idx = 0

        self.node_dict = defaultdict(Node)
        self.theme = THEMES[theme_color]
        self.node_shape = node_shape

        self.dot = Digraph()
        self.dot.attr("graph", 
                 bgcolor=self.theme["background_color"],
                 color=self.theme["outline_color"],
                 fontsize=self.theme["font_size"],
                 fontcolor=self.theme["font_color"],
                 fontname=self.theme["font_name"],
                 margin=self.theme["margin"],
                 pad=self.theme["padding"])
        self.dot.attr("node", shape=self.node_shape, 
                 style="filled", margin="0,0",
                 fillcolor=self.theme["fill_color"],
                 color=self.theme["outline_color"],
                 fontsize=self.theme["font_size"],
                 fontcolor=self.theme["font_color"],
                 fontname=self.theme["font_name"])
        self.dot.attr("edge", style="solid", 
                 color=self.theme["outline_color"],
                 fontsize=self.theme["font_size"],
                 fontcolor=self.theme["font_color"],
                 fontname=self.theme["font_name"])

        self.seen = set() # control depict
        self.mem_id_to_cus_id = dict() # control recursive

        if self.params is not None:
            assert all(isinstance(p, Variable) for p in self.params.values())
            self.param_map = {id(v): k for k, v in self.params.items()}
        
        self.make_dot()

    def size_to_str(self, size):
        return '\n(' + (', ').join(['%d' % v for v in size]) + ')'

    def make_dot(self):
        """ 
        Generates the graphviz object.
        """
        var = self.model(self.input)
        if isinstance(var, tuple):
            var = var[0]
        self._add_nodes(var.grad_fn)

    def _add_nodes(self, var, parent_id=None):

        cur_id = None

        if var not in self.seen:
            # add current node
            if torch.is_tensor(var):
                self.dot.node(str(id(var)), 
                              self.size_to_str(var.size()), 
                              fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = self.param_map[id(u)] if self.params is not None else ''
                node_name = '%s\n %s' % (name, self.size_to_str(u.size()))
                self.dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                cur_id = str(type(var).__name__) + str(self.layer_idx)
                self.dot.node(str(id(var)), cur_id)
                ### add layer info
                # id
                self.node_dict[cur_id].id = cur_id
                self.mem_id_to_cus_id[str(id(var))] = cur_id
                # type
                self.node_dict[cur_id].type = str(type(var).__name__)
                # parent
                if (parent_id is not None) and (parent_id is not ''):
                    self.node_dict[cur_id].parents.append(parent_id)
                self.layer_idx += 1

            self.seen.add(var)

            # visit children
            if hasattr(var, 'next_functions'):
                # obtain parameter shape
                for u in var.next_functions:
                    if (u[0] is not None) and (torch.is_tensor(u[0]) == False)\
                        and (hasattr(u[0], 'variable')):
                        assert cur_id is not None, 'bug'
                        self.node_dict[cur_id].param_shapes.append(u[0].variable.size())
                # obtain child_id
                for u in var.next_functions:
                    if u[0] is not None:
                        # connect with current node
                        self.dot.edge(str(id(u[0])), str(id(var)))
                        # append children id
                        if (torch.is_tensor(u[0]) == False) and\
                            (hasattr(u[0], 'variable') == False):
                            if u[0] not in self.seen:
                                child_id = str(type(u[0]).__name__) + str(self.layer_idx)
                            else:
                                child_id = self.mem_id_to_cus_id[str(id(u[0]))]
                            assert cur_id is not None, 'bug'
                            self.node_dict[cur_id].children.append(child_id)

                        self._add_nodes(var=u[0], parent_id=cur_id)

            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    self.dot.edge(str(id(t)), str(id(var)))
                    self._add_nodes(t)
        else:
            if (torch.is_tensor(var) == False) and\
                (hasattr(var, 'variable') == False):
                cur_id = self.mem_id_to_cus_id[str(id(var))]
                ## add layer info
                assert (parent_id is not None) and (parent_id is not '')
                # parent
                if (parent_id is not None) and (parent_id is not ''):
                    self.node_dict[cur_id].parents.append(parent_id)

    def view(self):
        """
        Saves the source code of self.dot to file and opens the
        rendered result in a viewer.
        """
        self.dot.view()

    def __getitem__(self, key):
        if isinstance(key, list):
            return [self.node_dict.get(k) for k in key]
        else:
            return self.node_dict.get(key)


class Node(object):
    """ 
    Node object for the Graph.

    """
    def __init__(self):
        self.parents = list()
        self.children = list()
        self.param_shapes = list()
        self.input_shapes = list()
        self.output_shapes = list()
        self.id = ''
        self.type = ''

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
    logger.info(f"Saved visualization at {path}")


def get_save_path(path, vis_type):
    """Return a save_path string."""
    path = "{}{}_{date:%Y-%m-%d_%H:%M:%S}.png".format(
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
        raw_field = layer._kernel._parameters['weight'].detach().numpy()
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
