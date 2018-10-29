"""Contains all visualization utilities."""
import numpy as np
import torch
from torch.nn import ReLU, SELU
import logging
import collections

logger = logging.getLogger(__name__)
# from ..models.basenetwork import BaseNetwork


def get_notable_indices(feature_importances, top_k=5):
    """
    Return dict of top k and bottom k features useful from matrix.

    Parameters
    ----------
    feature_importance : numpy.ndarray
        1D numpy array to extract the top of bottom indices.
    top_k : int
        How many features from top and bottom to extract.
        Defaults to 5.

    Returns
    -------
    notable_indices : dict
        Indices of the top most important features.
        Indices of the bottom mos unimportant features.

    """
    important_features = feature_importances.argsort()[-top_k:][::-1]
    unimportant_features = feature_importances.argsort()[:-1][:top_k]
    return {'important_indices': important_features,
            'unimportant_indices': unimportant_features}


class GuidedBackprop():
    """
    Generate gradients with guided back propagation w.r.t given input.

    Modified from https://github.com/utkuozbulak/pytorch-cnn-visualizations
    Insert backward hooks for activations to propagate positive gradients.

    Parameters
    ----------
    network : BaseNetwork
        Network to conduct guided backprop on.

    Returns
    -------
    gradients : numpy.ndarray
        Gradients of top most layer w.r.t the input sample.

    """

    def __init__(self, network):
        """Set up hooks for activations and gradient retrieval."""
        if network.__class__.__bases__[0].__name__ != "BaseNetwork":
            raise ValueError("Network type must be a subclass of BaseNetwork")
        self.network = network
        self.gradients = []
        self.hooks = []
        # Put model in evaluation mode
        self.network.eval()
        self._crop_negative_gradients()
        self._hook_top_layers()

    def _hook_top_layers(self):
        def hook_function(module, grad_in, grad_out):
            # TODO: Revisit dim disorder and check isinstance for classes.
            if isinstance(module, torch.nn.Linear):
                # grad_in shape is (bias, input, weights)
                self.gradients.append(grad_in[1])
            elif isinstance(module, torch.nn.modules.conv._ConvNd):
                # grad_in shape is (input, weights, bias)
                self.gradients.append(grad_in[0])
            else:
                raise NotImplementedError(
                    "{} module not available"
                    " for gradient extraction".format(type(module)))

        def get_top_layers(network):
            """In-order traversal to get only top layers of network."""
            all_top_layers = []
            if network.input_networks is not None:
                for net in network.input_networks:
                    all_top_layers.append(get_top_layers(net))
            else:
                all_top_layers.append(network.network[0]._kernel)
            return all_top_layers

        top_layers = get_top_layers(self.network)

        def flatten_list(l):
            """Flatten arbitrarily nested lists to get just the top layers."""
            if isinstance(l, collections.Iterable):
                return [itm for sublist in l for itm in flatten_list(sublist)]
            else:
                return [l]

        top_layers = flatten_list(top_layers)
        # Extract only unique top layers
        # This comes in handy if you have several inputs from the same network
        # at different depths and not return duplicated gradients.
        unique_top_layers = []
        for layer in top_layers:
            if layer not in unique_top_layers:
                unique_top_layers.append(layer)

        # Register hook to the first layers only
        for layer in unique_top_layers:
            self.hooks.append(layer.register_backward_hook(hook_function))

    def _crop_negative_gradients(self):
        """Update relu/selu activations to return positive gradients."""
        def activation_hook_function(module, grad_in, grad_out):
            """If there is a negative gradient, changes it to zero."""
            if isinstance(module, ReLU) or isinstance(module, SELU):
                return (torch.clamp(grad_in[0], min=0.0),)

        def hook_all_networks(network):
            self.hooks.append(
                    network.network[0]._activation.
                    register_backward_hook(activation_hook_function))
            logging.info("Cropping gradients in {}.".format(network.name))
            if network.input_networks is not None:
                for net in network.input_networks:
                    hook_all_networks(net)

        hook_all_networks(self.network)

    def _remove_hooks(self):
        """Remove all previously placed hooks from model."""
        for h in self.hooks:
            h.remove()

    def generate_gradients(self, input_data, targets):
        """
        Compute guided backprop gradients and returns top layer gradients.

        Parameters
        ----------
        input_data : numpy.ndarray or torch.Tensor
            1D for DenseNet, 4D (for 2D images) or 5D (for 3D images) Tensor.
        targets : numpy.ndarray or torch.LongTensor
            1D list of class labels

        Returns
        -------
        gradients : numpy.ndarray
            Gradient numpy array with same shape as input images.

        """
        if not isinstance(input_data, list):
            input_data = [input_data]
        # To properly pass the gradients
        for idx, t in enumerate(input_data):
            if not isinstance(t, torch.Tensor):
                input_data[idx] = torch.tensor(t, requires_grad=True)
            else:
                if not t.requires_grad:
                    t.requires_grad = True

        if not isinstance(targets, torch.LongTensor):
            targets = torch.LongTensor(targets)
        # Forward pass
        network_output = self.network.cpu()(input_data)
        # Zero gradients
        self.network.zero_grad()
        # Target for backprop
        one_hot_zeros = torch.zeros(
            network_output.size()[0],
            self.network._num_classes)
        one_hot_output = one_hot_zeros.scatter_(1, targets.unsqueeze(dim=1), 1)
        # Backward pass
        network_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # Will return batch dimension as well
        gradients_as_arr = \
            [grad.data.numpy() for grad in reversed(self.gradients)]
        self._remove_hooks()
        return gradients_as_arr
