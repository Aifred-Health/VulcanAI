"""Contains all visualization utilities."""
import numpy as np
import torch
from torch.nn import ReLU, SELU, ModuleList
import logging
import collections

logger = logging.getLogger(__name__)
# from ..models.basenetwork import BaseNetwork


def get_notable_indices(feature_importances, top_k=5):
    """
    Return dict of top k and bottom k features useful from matrix.

    Parameters:
        feature_importance : numpy.ndarray
            1D numpy array to extract the top of bottom indices.
        top_k : int
            How many features from top and bottom to extract.
            Defaults to 5.

    Returns:
        notable_indices : dict
            Indices of the top most important features.
            Indices of the bottom mos unimportant features.

    """
    important_features = feature_importances.argsort()[-top_k:][::-1]
    unimportant_features = feature_importances.argsort()[:-1][:top_k]
    return {'important_indices': important_features,
            'unimportant_indices': unimportant_features}


class GuidedBackprop(object):
    """
    Generate gradients with guided back propagation w.r.t given input.

    Modified from https://github.com/utkuozbulak/pytorch-cnn-visualizations
    Insert backward hooks for activations to propagate positive gradients.

    Parameters:
        network : BaseNetwork
            Network to conduct guided backprop on.

    Returns:
        gradients : list of numpy.ndarray
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

    def _crop_negative_gradients(self):
        """Update relu/selu activations to return positive gradients."""
        def activation_hook_function(module, grad_in, grad_out):
            """If there is a negative gradient, changes it to zero."""
            if isinstance(module, ReLU) or isinstance(module, SELU):
                return (torch.clamp(grad_in[0], min=0.0),)
            else:
                raise NotImplementedError("Only ReLU and SELU supported.")

        def hook_all_networks(network):
            self.hooks.append(
                    network.network[0]._activation.
                    register_backward_hook(activation_hook_function))
            logging.info("Cropping gradients in {}.".format(network.name))
            if network.input_networks:
                for in_net in network.input_networks.values():
                    hook_all_networks(in_net)
        # For Snapshot Networks
        if isinstance(self.network.network, ModuleList):
            for net in self.network.network:
                hook_all_networks(net)
        else:
            hook_all_networks(self.network)

    def _remove_hooks(self):
        """Remove all previously placed hooks from model."""
        for h in self.hooks:
            h.remove()

    def generate_gradients(self, input_data, targets):
        """
        Compute guided backprop gradients and returns top layer gradients.

        Parameters:
            input_data : numpy.ndarray or torch.Tensor
                2D for DenseNet, 4D (for 2D images) or 5D (for 3D images) Tensor.
            targets : numpy.ndarray or torch.LongTensor
                1D list of class labels

        Returns:
            gradients : list of numpy.ndarray
                Gradient list of numpy array with same shape as inputs.

        """
        assert isinstance(targets, torch.LongTensor)

        def requires_grad_multidataset(data_list):
            for d in data_list:
                if isinstance(d, list):
                    requires_grad_multidataset(d)
                else:
                    assert isinstance(d, torch.Tensor)
                    d.requires_grad_()

        if isinstance(input_data, list):
            requires_grad_multidataset(input_data)
        else:
            assert isinstance(input_data, torch.Tensor)
            input_data.requires_grad_()

        # Forward pass
        network_output = self.network(input_data)
        # Zero gradients
        self.network.zero_grad()
        # Target for backprop
        one_hot_zeros = torch.zeros(
            network_output.size()[0],
            self.network._num_classes)
        one_hot_output = one_hot_zeros.scatter_(1, targets.unsqueeze(dim=1), 1)
        one_hot_output = one_hot_output.to(self.network.device)
        # Backward pass
        network_output.backward(gradient=one_hot_output)

        def extract_input_gradients_multidataset(input_data):
            for data in input_data:
                if isinstance(data, list):
                    extract_input_gradients_multidataset(data)
                else:
                    self.gradients.append(data.grad.detach().cpu().numpy())

        if isinstance(input_data, list):
            extract_input_gradients_multidataset(input_data)
        else:
            self.gradients.append(input_data.grad.detach().cpu().numpy())

        self._remove_hooks()
        return self.gradients
