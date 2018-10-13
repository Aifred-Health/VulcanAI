"""Contains all visualization utilities."""
import torch
from torch.nn import ReLU, SELU
# from ..models.basenetwork import BaseNetwork

class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given input.
       Modified from https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/guided_backprop.py

       Insert backward hooks for all activations to propagate only positive gradients.
       Returns gradients of top most layer

       :param network: Network subclassed from BaseNetwork to conduct guided backprop.
       :param gradients: Hold gradients from top layer
       :param hooks: list of references to all hooks placed for removal after
    """
    def __init__(self, network):
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
        """
            Updates relu/selu activation functions so that it
            only returns positive gradients
        """
        def activation_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, changes it to zero
            """
            if isinstance(module, ReLU) or isinstance(module, SELU):
                return (torch.clamp(grad_in[0], min=0.0),)
        # Since all layer activations in a Net object point to the same
        # function we only need to hook one of them with
        # activation_hook_function
        if '_input_network' in self.network._modules:
            self.hooks.append(self.network._input_network.network[0]._activation.
                register_backward_hook(activation_hook_function))
        self.hooks.append(self.network.network[0]._activation.
            register_backward_hook(activation_hook_function))


    def _remove_hooks(self):
        """
        Remove all previously placed activation hooks from model.
        :return: None
        """
        for h in self.hooks:
            h.remove()

    def generate_gradients(self, input_data, targets):
        """
        Computes guided backprop gradients and returns top layer gradients.

        :param input_data: 1D for DenseNet, 4D (for 2D images) or 5D (for 3D images) Tensor.
        :param targets: 1D list of class truths of type torch.LongTensor
        :return: Gradient numpy array with same shape as input images
        """
        self.network.eval()
        # To properly pass the gradients
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(input_data, requires_grad=True)
        else:
            if not input_data.requires_grad:
                input_data.requires_grad = True
        # Forward pass
        network_output = self.network.cpu()(input_data)
        # Zero gradients
        self.network.zero_grad()
        # Target for backprop
        one_hot_zeros = torch.zeros(network_output.size()[0], network_output.size()[-1])
        one_hot_output = one_hot_zeros.scatter_(1, targets.unsqueeze(dim=1), 1)
        # Backward pass
        network_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # Will return batch dimension as well
        gradients_as_arr = self.gradients.data.numpy()
        self._remove_hooks()
        return gradients_as_arr
