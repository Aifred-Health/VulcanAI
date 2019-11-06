# -*- coding: utf-8 -*-
"""
This file defines a set of functions useful for working with pandas
dataframes full of tabular data
"""

import numpy as np
import pandas as pd
import logging
from itertools import groupby
from sklearn import preprocessing
import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


def convert_to_tensor_datasets(df, target_vars=None, continuous_target=False):
    """
    Given a df, returns a TensorDataset, with the target variables contained
    in the second tensor being specified by the column name(s) in target_vars.
    Args:
        df: Dataframe
            The dataframe to be operated on
        target_vars: string or list of string
            The column(s) to be used in the target tensor.
        continuous_target: boolean default False
            Whether the target values are continuous as opposed to categorical

    Returns: TensorDataset
        The resulting dataset representing the data contained in the df

    """

    if not target_vars:
        return TensorDataset(torch.Tensor(np.array(df)))

    if target_vars and not isinstance(target_vars, list):
        target_vars = [target_vars]

    data = torch.Tensor(np.array(df.drop(target_vars, axis=1)))

    if continuous_target:
        target = torch.FloatTensor(np.array(df[target_vars]))
    else:
        target = torch.LongTensor(np.array(df[target_vars]))

    dataset = TensorDataset(data, target)

    return dataset


# TODO: use kwargs
def create_label_encoding(df, column_name, ordered_values):
    """
    Create label encoding for the given column

    Parameters:
        df: Dataframe
            The dataframe to be manipulated
        column_name: String
            The name of the column you want encoded
        ordered_values: List or Dict
            Either an ordered list of possible column values.
            Or a mapping of column value to label value.
            Must include all possible values

    Returns:
        df
    """
    if isinstance(ordered_values, list):
        ordered_values = dict(map(lambda t: (t[1], t[0]), enumerate(
            ordered_values)))
    elif not isinstance(ordered_values, dict):
        raise ValueError("Must be either a list or dictionary")

    column_vals = ordered_values.keys()

    if len(column_vals) != len(set(column_vals)):
        raise ValueError("Ordered_value_list contains non-unique values")

    all_column_values = list(getattr(df, column_name).unique())
    if set(column_vals) != set(all_column_values):
        raise ValueError("Not all column values are included")

    df[column_name] = getattr(df, column_name).map(
        ordered_values)

    # logger.info("Successfully remapped %s", column_name)

    return df


def create_one_hot_encoding(df, column_name, prefix_sep="@"):
    """
    Create one-hot encoding for the given column.
    Parameters:
        column_name: String
            The name of the column you want to one-hot encode
        prefix_sep: String default("@")
            The prefix used when creating a one-hot encoding

    Returns:
        df: Dataframe
            The dataframe with the one hot encoding of the column added
    """
    # TODO: ensure dummy_na =False is what you want
    df = pd.get_dummies(df, columns=[column_name],
                             prefix_sep=prefix_sep)
    logger.info("Successfully encoded %s", column_name)

    return df

# if a use case presents itself column_name could easily become a list
def reverse_create_one_hot_encoding(df, prefix_sep, column_name=None):
    """
    Undo the creation of one-hot encodings, if prefix_sep was used
    to create one-hot encodings and nowhere else. If a column_name is
    provided, only that column will be reverse-encoded, otherwise all will
    be reverse-encoded.

    Parameters:
        df: Dataframe
            The dataframe to be manipulated
        column_name: String
            The name of the column to reverse the one-hot encoding for
        prefix_sep: String default("@")
            The prefix used when creating a one-hot encoding

    Returns:
        df
    """
    result_series = {}

    if column_name:
        if prefix_sep in column_name:
            considered_column_list = [column_name]
        else:
            raise ValueError("Prefix not found in column_name")
        non_dummy_cols = [col for col in df.columns
                          if col not in considered_column_list]
    else:
        considered_column_list = df.columns
        non_dummy_cols = [col for col in df.columns
                          if prefix_sep not in col]

    # Find dummy columns and build pairs (category, category_value)
    dummy_tuples = [(col.split(prefix_sep)[0], col) for col in
                    considered_column_list if prefix_sep in col]

    for dummy, cols in groupby(dummy_tuples, lambda item: item[0]):
        dummy_df = df[[col[1] for col in cols]]

        # Find max value among columns
        max_columns = dummy_df.idxmax(axis=1)

        # Remove category_ prefix
        result_series[dummy] = max_columns.apply(lambda item:
                                                 item.split(prefix_sep)[1])

    # Copy non-dummy columns over.
    for col in non_dummy_cols:
        result_series[col] = df[col]

    df = pd.DataFrame(result_series)

    # logger.info("Successfully converted %d \
    #                columns back from dummy format.", len(dummy_tuples))

    return df


def identify_null(df, threshold):
    """
    Return columns where there is at least threshold percent of
    null values.

    Parameters:
        df: Dataframe
            The dataframe to be manipulated
        threshold: Float
            A number between 0 and 1, representing proportion of null
            values.

    Returns:
        cols: List
            A list of those columns with threshold percentage null values

    """
    if threshold >= 1 or threshold <= 0:
        raise ValueError(
            "Threshold needs to be a proportion between 0 and 1 \
            (exclusive)")
    num_threshold = ((1 - threshold) * len(df))
    # thresh is "Require that many non-NA values."
    tmp = df.dropna(thresh=num_threshold, axis=1)
    cols = list(set(df.columns).difference(set(tmp.columns)))
    return cols


def identify_unique(df, threshold):
    """
    Returns columns that do not have have at least threshold number of
    values. If a column has 9 values and the threshold is 9, that column
    will not be returned.

    Parameters:
        df: Dataframe
            The dataframe to be manipulated
        threshold: The minimum number of values needed.
            Must be greater than 1. Not between 0 and 1, but rather
            represents the number of values

    Returns:
        column_list: list
            The list of columns having threshold number of values

    """
    obj_types = {col: set(map(type, df[col])) for col in
                 df.columns}

    column_list = []
    for col in df.columns:
        if len(obj_types[col]) > 1:
            logger.warning("Column: {} has mixed datatypes, this may"
                           "interfere with an accurate identification"
                           "of mixed values: i.e. you may have 1 and '1'"
                           .format(col))
        if len(df[col].unique()) < threshold:
            column_list.append(col)
    return column_list


def identify_unbalanced_columns(df, threshold, non_numeric=True):
    """
    This returns columns that are highly unbalanced.
    Those that have a disproportionate amount of one value.

    Parameters:
        df: Dataframe
            The dataframe to be manipulated
        threshold: Float
            Proportion needed to define unbalanced, between 0 and 1
            0 is a lesser proportion of the one value,
            (less imbalanced)
        non_numeric: Boolean
            Whether non-numeric columns are also considered.

    Returns:
        column_list: List
            The list of column names

    """
    if non_numeric:
        columns = list(df.columns)
    else:
        columns = list(df.select_dtypes(include=np.number))
    column_list = []
    for col in columns:
        # Check amount of null values, because if column is entirely null,
        # max won't work.
        num_of_null = df[col].isnull().sum()
        if num_of_null != len(df):
            col_maj = (max(df[col].value_counts()) /
                       df[col].value_counts().sum())
            if col_maj >= threshold:
                column_list.append(col)
    return column_list


def identify_highly_correlated(df, threshold):
    """
    Identify columns that are highly correlated with one-another.

    Parameters:
        df: Dataframe
            The dataframe to be manipulated
        threshold: Between 0 (weakest correlation) and 1
            (strongest correlation).
            Minimum amount of correlation necessary to be identified.

    Returns:
        column list: List of tuples
            The correlation values above threshold

    """
    column_list = set()
    features_correlation = df.corr().abs()
    corr_pairs = features_correlation.unstack()
    for index, val in corr_pairs.items():
        if val > threshold and (index[0] != index[1]):
            column_list.add((index, val))
    return column_list


def identify_low_variance(df, threshold):
    """
    Identify those columns that have low variance

    Parameters:
        df: Dataframe
            The dataframe to be manipulated
        threshold: Float
            Between 0 an 1.
            Maximum amount of variance necessary to be identified as low
            variance.

    Returns:
        variance_dict: Dict
            A dictionary of column names, with the value being their
            variance.

    """
    dct_low_var = {}
    scaler = preprocessing.MinMaxScaler()
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            col_float_array = df[col].dropna().values.astype(float)
            reshaped_col_float_array = col_float_array.reshape(-1, 1)
            scaled_col = scaler.fit_transform(reshaped_col_float_array)
            col_var = scaled_col.var()
            if col_var <= threshold:
                dct_low_var[col] = col_var
    return dct_low_var


def convert_all_categorical_binary(df, list_only=False,
                                   exception_columns=None):
    """Recodes all columns with only two values as ones and zeros, float
    valued.

    This should only be used once you have a final dataset in case values
    are not actually binary. Useful for dealing with all the variations of
    YES/NO ,yEs/nO etc.

    Parameters:
        df: Dataframe
            The dataframe to be manipulated
        list_only: boolean
            only return a list of columns for which this would
            apply, do not actually do the transformation.
            If false operation is performed in place.
        exception_columns: list
            list of column names you do not wish to convert.

    Returns:
        list or df
        list if list_only if true, nothing otherwise.
    """

    # recoding binary valued columns as ones and zeros

    binary_cols = [(col, df[col].value_counts().index) for col
                   in df.columns if
                   len(df[col].value_counts()) == 2]

    if list_only:
        return binary_cols

    for col, index in binary_cols:
        if col in exception_columns:
            continue
        di = {index[0]: 1.0, index[1]: 0.0}
        try:
            df = df.replace({col: di})
        except (AssertionError, TypeError, ValueError) as e:
            logging.info("Could not convert column {} due to error {}"
                         .format(col, e))
            continue
        df[col] = df[col].astype(np.float64)

    return df


def stitch_datasets(df_main=None, merge_on_columns=None,
                    index_list=None, **dataset_dict):
    """
    Function to produce a single dataset from multiple.

    Parameters:
        df_main: dataframe
            optional primary dataframe to merge onto
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
    >>> df_stitched = stitch_datasets(merge_on_columns=['A'], index_list=['A'],
     df1=df_test_one, df2=df_test_two)

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
        df_group_on = merged_df.reset_index(drop=True).\
            groupby(merge_on_columns).apply(lambda x: x.bfill().ffill())
        logger.info("\tDropping duplicates")

        # Drop rows where there are duplicates for the merged_on_columns.
        # We first need to dropna based on merged since drop_duplicates
        # ignores null/na values.
        df_group_on = df_group_on.dropna(subset=merge_on_columns, how='all')
        df_group_on = df_group_on.drop_duplicates(subset=merge_on_columns,
                                                  keep='first', inplace=False)
        merged_df = df_group_on

    if index_list is not None:
        merged_df = merged_df.set_index(index_list, inplace=False)

    logger.info("\nMerge Total columns = {totalCols}, rows = {totalRows} "
                .format(totalCols=len(list(merged_df)),
                        totalRows=len(merged_df)))
    return merged_df