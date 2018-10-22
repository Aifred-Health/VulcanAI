# -*- coding: utf-8 -*-
"""
This file contains utility methods that many be useful to several dataset classes.
check_split_ration, stratify, rationed_split, randomshuffler
were all copy-pasted from torchtext because torchtext is not yet packaged
for anaconda and is therefore not yet a reasonable dependency.
See https://github.com/pytorch/text/blob/master/torchtext/data/dataset.py
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def check_split_ratio(split_ratio):
    """
    Check that the split ratio argument is not malformed
    :param split_ratio: desired split ratio, either a list of length 2 or 3 depending if the validation set is desired.
    :return: split ratio as tuple
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
            "Length of split ratio list should be 2 or 3, got {}".format(split_ratio))

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

# def stratify(examples, strata_field):
#
#     # The field has to be hashable otherwise this doesn't work
#     # There's two iterations over the whole dataset here, which can be
#     # reduced to just one if a dedicated method for stratified splitting is used
#     unique_strata = set(getattr(example, strata_field) for example in examples)
#     strata_maps = {s: [] for s in unique_strata}
#     for example in examples:
#         strata_maps[getattr(example, strata_field)].append(example)
#     return list(strata_maps.values())


# def rationed_split(examples, train_ratio, test_ratio, val_ratio, rnd):
#     """
#     Create a random permutation of examples, then split them
#     by ratio x length slices for each of the train/test/dev? splits
#     :param examples:
#     :param train_ratio: Train ratio
#     :param test_ratio: Test ratio
#     :param val_ratio: Val ratio
#     :param rnd:
#     :return:
#     """
#     # Create a random permutation of examples, then split them
#     # by ratio x length slices for each of the train/test/dev? splits
#     N = len(examples)
#     randperm = rnd(range(N))
#     train_len = int(round(train_ratio * N))
#
#     # Due to possible rounding problems
#     if not val_ratio:
#         test_len = N - train_len
#     else:
#         test_len = int(round(test_ratio * N))
#
#     indices = (randperm[:train_len],  # Train
#                randperm[train_len:train_len + test_len],  # Test
#                randperm[train_len + test_len:])  # Validation
#
#     # There's a possibly empty list for the validation set
#     data = tuple([examples[i] for i in index] for index in indices)
#
#     return data

# class RandomShuffler(object):
#     """Use random functions while keeping track of the random state to make it
#     reproducible and deterministic."""
#
#     def __init__(self, random_state=None):
#         self._random_state = random_state
#         if self._random_state is None:
#             self._random_state = random.getstate()
#
#     @contextmanager
#     def use_internal_state(self):
#         """Use a specific RNG state."""
#         old_state = random.getstate()
#         random.setstate(self._random_state)
#         yield
#         self._random_state = random.getstate()
#         random.setstate(old_state)
#
#     @property
#     def random_state(self):
#         return deepcopy(self._random_state)
#
#     @random_state.setter
#     def random_state(self, s):
#         self._random_state = s
#
#     def __call__(self, data):
#         """Shuffle and return a new list."""
#         with self.use_internal_state():
#             return random.sample(data, len(data))


# THIS IS FROM SNEHA  https://github.com/sneha-desai
# TODO: replace with Joseph's version
def stitch_datasets(df_list, on, index_list=None):
    """
    Args:
    df_list: list of dataframes to stitch together
    on: key that specifies which features column to use in each dataset
    to identify the specific examples of all datasets
    index_list: list of feature columns to add present bit (default None)

    Returns: concatenated dataframe

    """
    print(index_list)
    first_column = list(df_list)[1]
    merged_df = df_list[first_column].copy(deep=True)
    merged_df = merged_df.apply(pd.to_numeric, errors='ignore')

    count = 1

    for key in list(df_list):
        if key != first_column:
            print('Combining: {} '.format(key))

            count = count + 1
            df_two = df_list[key].copy(deep=True)
            df_two = df_two.apply(pd.to_numeric, errors='ignore')
            merged_df = merged_df.append(df_two)

    if on is not None:
        # Group by keys, forward fill and backward fill missing data then remove duplicate keys
        df_groupOn = merged_df.reset_index().groupby(on).apply(
            lambda x: x.fillna(method='ffill').fillna(method='bfill'))
        if 'index' in list(df_groupOn):
            del df_groupOn['index']
        print("\tDropping duplicates")

        merged_df = df_groupOn.drop_duplicates(subset=on)

    print("\nMerge Total columns = {totalCols}, rows = {totalRows} ".format(
        totalCols=len(list(merged_df)),
        totalRows=len(merged_df)))
    return merged_df

