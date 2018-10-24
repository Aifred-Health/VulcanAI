"""Define utilities for all networks."""
import torch
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from collections import OrderedDict as odict


def round_list(raw_list, decimals=4):
    """
    Return the same list with each item rounded off.

    Parameters
    ----------
    raw_list : float list
        float list to round.
    decimals : int
        How many decimal points to round to.

    Returns
    -------
    rounded_list : float list
        The rounded list in the same shape as raw_list.

    """
    return [round(item, decimals) for item in raw_list]


def get_confusion_matrix(predictions, targets):
    """
    Calculate the confusion matrix for classification network predictions.

    Parameters
    ----------
    predictions : numpy.ndarray
        The classes predicted by the network. Does not take one hot vectors.
    targets : numpy.ndarray
        the classes of the ground truth. Does not take one hot vectors.

    Returns
    -------
    confusion_matrix : numpy.ndarray
        The confusion matrix.

    """
    if len(predictions.shape) == 2:
        predictions = predictions[:, 0]
    if len(targets.shape) == 2:
        targets = targets[:, 0]
    return confusion_matrix(y_true=targets,
                            y_pred=predictions)


def get_one_hot(in_matrix):
    """
    Reformat truth matrix to same size as the output of the dense network.

    Parameters
    ----------
    in_matrix : numpy.ndarray
        The categorized 1D matrix

    Returns
    -------
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

def get_size(summary_dict, output):
    """
    Helper function for the BaseNetwork's get_output_shapes
    """
    if isinstance(output, tuple):
        for i in range(len(output)):
            summary_dict[i] = odict()
            summary_dict[i] = get_size(summary_dict[i], output[i])
    else:
        summary_dict['output_shape'] = list(output.size())
    return summary_dict


def pad(tensor, padded_shape):
    """
    Pad incoming tensor to the size of padded_shape.

    tensor must have same spatial dimenison as the padded_shape.
    Useful for combining various conv dimension outputs and to implement
    'same' padding for conv operations.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to be padded
    padded_shape : np.array
        Final padded tensor shape [*spatial_dimensions]

    Returns
    -------
    tensor : torch.Tensor
        zero padded tensor with spatial dimension as padded_shape

    """
    # Ignore channels and batch and focus on spatial dimensions
    # from incoming tensor
    if not isinstance(padded_shape, np.ndarray):
        padded_shape = np.array(padded_shape)
    n_dim = len(padded_shape)
    # Calculate, element-wise, how much needs to be padded for each dim.
    dims_size_diff = padded_shape - np.array(tensor.shape[-n_dim:])
    # TODO: Use torch.nn.ConstantPadding?
    padding_needed = []
    for dim_diff in reversed(dims_size_diff):
        dim_zero_padding = ceil(dim_diff/2)
        dim_one_padding = floor(dim_diff/2)
        padding_needed.append(dim_zero_padding)
        padding_needed.append(dim_one_padding)
    return F.pad(tensor, padding_needed)
