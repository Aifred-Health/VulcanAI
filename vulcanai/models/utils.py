# coding=utf-8
"""Define utilities for all networks."""
from math import ceil, floor
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import math
from sklearn.preprocessing import LabelBinarizer
from collections import OrderedDict
from collections import defaultdict


def _get_probs(network, loader, index_to_iter, ls_feat_vals):
    """Returns probability for each object within loader based on output
    from training neural network

    Parameters:
        network : vulcan.model
            training vulcan network
        loader : torch.dataloader
            dataloader containing validation set
        index_to_iter : string
            feature to iterate through adjusting values
        ls_feat_vals : list
            values to iterate through for feature in index_to_iter

    Returns:
        dct_scores : dictionary
            dictionary of scores
    """

    dct_scores = defaultdict()
    for index in range(len(loader)):
        dct_scores[index] = {}
    for index in range(len(loader)):
        # Extract specific index from loader and create a DataLoader instance
        # to send to forward_pass
        input_loader = DataLoader(TensorDataset(loader.dataset[index][0]
                                                .unsqueeze(0),
                                                loader.dataset[index][1]
                                                .unsqueeze(0)))

        subj_prob = network.forward_pass(data_loader=input_loader,
                                         transform_outputs=False)
        # Standardize probability of positive label
        subj_prob = subj_prob[0][1] * 100
        subj_prob = round(subj_prob, 2)
        # Add probability to scores dictionary where keys are the index
        # and value the probability belongs to.
        dct_scores[index][loader.dataset[index][0][index_to_iter].item()] = \
            subj_prob
        # Iterate through other possible values and find probability of
        # positive label and add to dictionary
        # for current index.
        for new_val in ls_feat_vals:
            if new_val != loader.dataset[index][0][index_to_iter].item():
                loader.dataset[index][0][index_to_iter] = new_val
                input_loader = DataLoader(TensorDataset(loader
                                                        .dataset[index][0]
                                                        .unsqueeze(0),
                                                        loader
                                                        .dataset[index][1]
                                                        .unsqueeze(0)))
                subj_prob = network.forward_pass(data_loader=input_loader,
                                                 ransform_outputs=False)
                subj_prob = subj_prob[0][1] * 100
                subj_prob = round(subj_prob, 2)
                dct_scores[index][new_val] = subj_prob
    return dct_scores


def _filter_matched_subj(dct_scores, loader, index_to_iter):
    """
    Returns dictionary of filtered keys based on predicted value and value
    truly assigned in actual set.

    Parameters:
        dct_scores : dictionary
            dictionary of probability scores produced in get_scores function
        loader : torch.dataloader
            dataloader containing validation set
        index_to_iter : string
            feature to iterate through adjusting values
    Returns:
        dct_filtered : dictionary
            dictionary of filtered entities 
    """
    dct_filtered = {}
    for subj in list(dct_scores):
        highest_prob = max(dct_scores[subj].values())
        highest_val = [val for val, prob in dct_scores[subj].items()
                       if prob == highest_prob]
        if loader.dataset.dataset[subj][0][index_to_iter].item() \
                in highest_val:
            dct_filtered[subj] = highest_prob
    return dct_filtered


def round_list(raw_list, decimals=4):
    """
    Return the same list with each item rounded off.

    Parameters:
        raw_list : float list
            float list to round.
        decimals : int
            How many decimal points to round to.

    Returns:
        rounded_list : float list
            The rounded list in the same shape as raw_list.

    """
    return [round(item, decimals) for item in raw_list]


def get_one_hot(in_matrix):
    """
    Reformat truth matrix to same size as the output of the dense network.

    Parameters:
        in_matrix : numpy.ndarray
            The categorized 1D matrix

    Returns:
        one_hot : numpy.ndarray
            A one-hot matrix representing the categorized matrix

    """
    if in_matrix.dtype.name == 'category':
        custom_array = in_matrix.cat.codes

    elif isinstance(in_matrix, np.ndarray):
        custom_array = in_matrix

    else:
        raise ValueError("Input matrix cannot be converted.")

    lb = LabelBinarizer()
    return np.array(lb.fit_transform(custom_array), dtype='float32')


def pad(tensor, target_shape):
    """
    Pad incoming tensor to the size of target_shape.

    tensor must have same spatial dimenison as the target_shape.
    Useful for combining various conv dimension outputs and to implement
    'same' padding for conv operations.

    Parameters:
        tensor : torch.Tensor
            Tensor to be padded
        target_shape : np.array
            Final padded tensor shape [*spatial_dimensions]

    Returns:
        tensor : torch.Tensor
            zero padded tensor with spatial dimension as target_shape

    """
    # Ignore channels and batch and focus on spatial dimensions
    # from incoming tensor
    if not isinstance(target_shape, np.ndarray):
        target_shape = np.array(target_shape)
    n_dim = len(target_shape)
    # Calculate, element-wise, how much needs to be padded for each dim.
    dims_size_diff = target_shape - np.array(tensor.shape[-n_dim:])
    # TODO: Use torch.nn.ConstantPadding?
    padding_needed = []
    for dim_diff in reversed(dims_size_diff):
        dim_zero_padding = ceil(dim_diff/2)
        dim_one_padding = floor(dim_diff/2)
        padding_needed.append(dim_zero_padding)
        padding_needed.append(dim_one_padding)
    return f.pad(tensor, padding_needed)


# noinspection PyTypeChecker
def network_summary(network, input_size=None):
    """
    Returns the summary of shapes of all layers in the network
    :return: OrderedDict of shape of each layer in the network
    """
    if not input_size:
        input_size = []
        for net in network.input_networks.values():
            input_size.append(net.in_dim)
    # input_size must be a list
    if isinstance(input_size, (tuple, int)):
        input_size = [input_size]
    
    def get_size(summary_dict, output):
        """
        Helper function for the BaseNetwork's get_output_shapes
        """
        if isinstance(output, tuple):
            for i in range(len(output)):
                summary_dict[i] = OrderedDict()
                summary_dict[i] = get_size(summary_dict[i], output[i])
        else:
            summary_dict['output_shape'] = tuple(output.size())
        return summary_dict

    # noinspection PyUnresolvedReferences,PyUnresolvedReferences
    def register_hook(module):
        """
        Registers a backward hook
        For more info: https://pytorch.org/docs/stable/_modules/torch/
        tensor.html#Tensor.register_hook

        """

        # noinspection PyShadowingNames, PyUnresolvedReferences
        def hook(module, input, output):
            """
            https://github.com/pytorch/tutorials/blob/8afce8a213cb3712aa7de
            1e1cf158da765f029a7/beginner_source/former_torchies/nn_tutorial
            .py#L146
            """
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)
            # Test
            m_key = '%s-%i' % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]['input_shape'] = tuple(input[0].size())
            summary[m_key] = get_size(summary[m_key], output)
            # Test
            params = 0
            if hasattr(module, 'weight'):
                params += torch.prod(torch.LongTensor(tuple(
                    module.weight.size())))
                if module.weight.requires_grad:
                    summary[m_key]['trainable'] = True
                else:
                    summary[m_key]['trainable'] = False
            if hasattr(module, 'bias'):
                params += torch.prod(torch.LongTensor(tuple(
                    module.bias.size())))
            # Test
            summary[m_key]['nb_params'] = params
        if not isinstance(module, torch.nn.Sequential) and \
                not isinstance(module, torch.nn.ModuleList) and \
                not (module == network):
            hooks.append(module.register_forward_hook(hook))

    x = []
    for in_size in input_size:
        x.append(torch.empty(1, *in_size))
    
    if len(x) == 1:
        x = torch.cat(x, dim=1)

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    network.apply(register_hook)
    # make a forward pass
    network.cpu()(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    return summary


def print_model_structure(network):
    """Print the entire model structure."""
    shapes = network_summary(network)
    for k, v in shapes.items():
        print('{}:'.format(k))
        if isinstance(v, OrderedDict):
            for k2, v2 in v.items():
                print('\t {}: {}'.format(k2, v2))


# noinspection PyProtectedMember
def selu_weight_init_(tensor, mean=0.0):
    """
    SELU layer weight initialization function.

    Function assigned to variable that will be called within
    _init_weights function to assign weights for selu.

    Parameters:
        tensor :  torch.tensor
            Weight tensor to be adjusted
        mean : float
            Mean value for the normal distribution

    Returns:
        torch.tensor
            weight tensor with normal distribution

    """
    with torch.no_grad():
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(tensor)
        std = math.sqrt(1. / fan_in)
        return nn.init.normal_(tensor, mean, std)


def selu_bias_init_(tensor, const=0.0):
    """
    SELU layer bias initialization function.

    Function assigned to variable that will be called within
    _init_bias function to assign bias for selu.

    Parameters:
        tensor : torch.tensor
            Bias tensor to be adjusted
        const : float
            Constant value to be assigned to tensor.

    Returns:
        torch.tensor
            bias tensor with constant values.

    """
    with torch.no_grad():
        return nn.init.constant_(tensor, const)


def set_tensor_device(data, device=None):
    """
    Convert list of data tensors to specified device.

    Parameters:
        data : torch.tensor or list
            data to be converted to the specified device.
        device : str or torch.device
            the desired device

    Returns:
        data : torch.tensor or list
            data converted to the specified device

    """
    if not isinstance(data, (list, tuple)):
        data = data.to(device=device, non_blocking=True)

    else:
        for idx, d in enumerate(data):
            data[idx] = set_tensor_device(d, device=device)
    return data


def master_device_setter(network, device=None):
    """
    Convert network and input_networks to specified device.

    Parameters:
        network : BaseNetwork
            network to be converted to the specified device.
        device : str or torch.device
            the desired device

    """
    network.device = device
    if network.input_networks:
        for net in network.input_networks.values():
            master_device_setter(net, device)
