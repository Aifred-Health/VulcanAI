import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer

def get_notable_indices(feature_importances, top_k=5):
    """
    Return dict of top k and bottom k features useful from matrix.

    :param feature_importance: 1D numpy array
    :param top_k: defaults to top and bottom 5 indices
    """
    important_features = feature_importances.argsort()[-top_k:][::-1]
    unimportant_features = feature_importances.argsort()[:-1][:top_k]
    return {'important_indices': important_features,
            'unimportant_indices': unimportant_features}


def round_list(raw_list, decimals=4):
    """
    Return the same list with each item rounded off.

    Args:
    :param raw_list: float list
    :param decimals: how many decimal points to round to
    :return: the rounded list in the same shape
    """
    return [round(item, decimals) for item in raw_list]


def get_confusion_matrix(predictions, targets):
    """
    Calculate the confusion matrix for classification network predictions.

    :param predictions: the class matrix predicted by the network.
                        oes not take one hot vectors.
    :param targets: the class matrix of the ground truth
                    Does not take one hot vectors.

    :return: the confusion matrix
    """
    print(type(predictions))
    if len(predictions.shape) == 2:
        predictions = predictions[:, 0]
    if len(targets.shape) == 2:
        targets = targets[:, 0]
    return confusion_matrix(y_true=targets,
                            y_pred=predictions)


def get_one_hot(in_matrix):
    """
    Reformat truth matrix to same size as the output of the dense network.

    Args:
        in_matrix: the categorized 1D matrix

    Returns: a one-hot matrix representing the categorized matrix
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