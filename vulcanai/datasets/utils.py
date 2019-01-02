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

# def stratify(examples, strata_field):
#
#     # The field has to be hashable otherwise this doesn't work
#     # There's two iterations over the whole dataset here, which can be
#     # reduced to just one if a dedicated method for stratified splitting is
# used
#     unique_strata = set(getattr(example, strata_field) for example in
# examples)
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
#     """Use random functions while keeping track of the random state to
# make it
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


def stitch_datasets(df_main=None, merge_on_columns=None,
                    index_list=None, **dataset_dict):
    """
    Function to produce a single dataset from multiple.

    Parameters:
        df_dict : dictionary of dataframes to concatenated
            dictionary {key = df name: value = dataframe} of dataframes to
            stitch together.
        merge_on_columns : list of strings
            key(s) that specifies which columns to use to uniquely stitch
            dataset (default None)
        index_list: list of strings
            columns to establish as index for final stitched dataset
            (default None)
        dataset_dict : keyword parameter, value is dataframe
            pandas dataframe assigned to keyword argument that produces a
            dictionary variable.

    Returns:
        merged_df : dataframe
            concatenated dataframe

    :Example:

    >>> df_test_one: pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                                   'B': ['B0', 'B1', 'B2', 'B3'],
                                   'C': ['C0', 'C1', 'C2', 'C3'],
                                   'D': ['D0', 'D1', 'D2', 'D3']},
                                   index=[0, 1, 2, 3]),

     >>> df_test_two: pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                                    'B': ['B4', 'B5', 'B6', 'B7'],
                                    'C': ['C4', 'C5', 'C6', 'C7'],
                                    'D': ['D4', 'D5', 'D6', 'D7']},
                                    index=[4, 5, 6, 7])}
    >>> df_stitched = stitch_datasets(merge_on_columns=['A'], index_list=['A'], df1=df_test_one, df2=df_test_two)

    """
    # Get name of a dataframe to extract
    if df_main is not None:
        merged_df = copy.deepcopy(df_main)
    else:
        first_column = list(dataset_dict)[0]
        merged_df = dataset_dict.pop(first_column)
        merged_df = merged_df.apply(pd.to_numeric, errors='ignore')

    for key in list(dataset_dict):
        logger.info('Combining: {}'.format(key))
        df_two = dataset_dict.pop(key)
        merged_df = pd.concat([merged_df, df_two], sort=False)

    if merge_on_columns is not None:
        # Group by keys, forward fill and backward fill missing data
        # then remove duplicate keys
        merged_df = merged_df.apply(pd.to_numeric, errors='ignore')
        df_groupOn = merged_df.reset_index(drop=True).\
            groupby(merge_on_columns).apply(lambda x: x.bfill().ffill())
        logger.info("\tDropping duplicates")

        #Drop rows where there are duplicates for the merged_on_columns.
        # We first need to dropna based on merged since drop_duplicates
        # ignores null/na values.
        df_groupOn = df_groupOn.dropna(subset=merge_on_columns, how='all')
        df_groupOn = df_groupOn.drop_duplicates(subset=merge_on_columns,
                                                keep='first', inplace=False)
        merged_df = df_groupOn

    if index_list is not None:
        merged_df = merged_df.set_index(index_list, inplace=False)

    logger.info("\nMerge Total columns = {totalCols}, rows = {totalRows} "
        .format(
        totalCols=len(list(merged_df)),
        totalRows=len(merged_df)))
    return merged_df
