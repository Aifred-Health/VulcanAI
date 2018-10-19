"""Contains all visualization utilities."""
import torch
from torch.nn import ReLU, SELU
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
        self.gradients = None
        self.hooks = []
        # Put model in evaluation mode
        self.network.eval()
        self._crop_negative_gradients()
        self._hook_top_layers()

    def _hook_top_layers(self):
        def hook_function(module, grad_in, grad_out):
            # TODO: Revisit dim disorder and check isinstance for classes.
            if module.__class__.__name__ == 'Linear':
                # grad_in shape is (bias, input, weights)
                self.gradients = grad_in[1]
            elif module.__class__.__bases__[0].__name__ == '_ConvNd':
                # grad_in shape is (input, weights, bias)
                self.gradients = grad_in[0]
        # Register hook to the first layer
        # TODO: Modify for multi-input NNs
        if '_input_network' in self.network._modules:
            first_layer = self.network._input_network.network[0]._kernel
        else:
            first_layer = self.network.network[0]._kernel
        self.hooks.append(first_layer.register_backward_hook(hook_function))

    def _crop_negative_gradients(self):
        """Update relu/selu activations to return positive gradients."""
        def activation_hook_function(module, grad_in, grad_out):
            """If there is a negative gradient, changes it to zero."""
            if isinstance(module, ReLU) or isinstance(module, SELU):
                return (torch.clamp(grad_in[0], min=0.0),)
        # Since all layer activations in a Net object point to the same
        # function we only need to hook one of them with
        # activation_hook_function
        if '_input_network' in self.network._modules:
            self.hooks.append(
                self.network._input_network.network[0].
                _activation.register_backward_hook(activation_hook_function))
        self.hooks.append(
            self.network.network[0].
            _activation.register_backward_hook(activation_hook_function))

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
        self.network.eval()
        # To properly pass the gradients
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(input_data, requires_grad=True)
        else:
            if not input_data.requires_grad:
                input_data.requires_grad = True
        if not isinstance(targets, torch.LongTensor):
            targets = torch.LongTensor(targets)
        # Forward pass
        network_output = self.network.cpu()(input_data)
        # Zero gradients
        self.network.zero_grad()
        # Target for backprop
        one_hot_zeros = torch.zeros(
            network_output.size()[0],
            network_output.size()[-1])
        one_hot_output = one_hot_zeros.scatter_(1, targets.unsqueeze(dim=1), 1)
        # Backward pass
        network_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # Will return batch dimension as well
        gradients_as_arr = self.gradients.data.numpy()
        self._remove_hooks()
        return gradients_as_arr
