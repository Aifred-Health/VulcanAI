# -*- coding: utf-8 -*-
"""
This file contains utility methods that many be useful to several dataset
classes.
check_split_ration, stratify, rationed_split, randomshuffler
were all copy-pasted from torchtext because torchtext is not yet packaged
for anaconda and is therefore not yet a reasonable dependency.
See https://github.com/pytorch/text/blob/master/torchtext/data/dataset.py
"""
import pandas as pd
import logging
import copy
import numpy as np

logger = logging.getLogger(__name__)


# TODO: implement
def clean_dataframe(df):
    """
    Goes through and ensures that all nonsensical values are encoded as NaNs
    :param df:
    :return:
    """

    return df


def check_split_ratio(split_ratio):
    """
    Check that the split ratio argument is not malformed

    Parameters:

        split_ratio: desired split ratio, either a list of length 2 or 3
            depending if the validation set is desired.
    
    Returns:
        split ratio as tuple

    """
    valid_ratio = 0.
    if isinstance(split_ratio, float):
        # Only the train set relative ratio is provided
        # Assert in bounds, validation size is zero
        assert 0. < split_ratio < 1., (
            "Split ratio {} not between 0 and 1".format(split_ratio))

        test_ratio = 1. - split_ratio
        return split_ratio, test_ratio, valid_ratio
    elif isinstance(split_ratio, list):
        # A list of relative ratios is provided
        length = len(split_ratio)
        assert length == 2 or length == 3, (
            "Length of split ratio list should be 2 or 3, got {}".format(
                split_ratio))

        # Normalize if necessary
        ratio_sum = sum(split_ratio)
        if not ratio_sum == 1.:
            split_ratio = [float(ratio) / ratio_sum for ratio in split_ratio]

        if length == 2:
            return tuple(split_ratio + [valid_ratio])
        return tuple(split_ratio)
    else:
        raise ValueError('Split ratio must be float or a list, got {}'
                         .format(type(split_ratio)))


def rationed_split(df, train_ratio, test_ratio, validation_ratio):
    """
    Function to split a dataset given ratios. Assumes the ratios given
    are valid (checked using check_split_ratio).

    Parameters:
        df: Dataframe
            The dataframe you want to split
        train_ratio: int
            proportion of the dataset that will go to the train split.
            between 0 and 1
        test_ratio: int
            proportion of the dataset that will go to the test split.
            between 0 and 1
        validation_ratio: int
            proportion of the dataset that will go to the val split.
            between 0 and 1

    Returns:
        indices: tuple of list of indices.
    """
    n = len(df.index)
    perm = np.random.permutation(df.index)
    train_len = int(round(train_ratio * n))

    # Due to possible rounding problems
    if not validation_ratio:
        test_len = n - train_len
    else:
        test_len = int(round(test_ratio * n))

    indices = (perm[:train_len],  # Train
               perm[train_len:train_len + test_len],  # Test
               perm[train_len + test_len:])  # Validation

    return indices



