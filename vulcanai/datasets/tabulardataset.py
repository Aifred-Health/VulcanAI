# -*- coding: utf-8 -*-
"""
This file defines the TabularDataset Class
"""
import torch
from torch.utils.data import Dataset
import copy
import numpy as np
import pandas as pd
from . import utils as utils
import logging
from itertools import groupby
from sklearn import preprocessing
import time

logger = logging.getLogger(__name__)


# noinspection PyCallingNonCallable
class TabularDataset(Dataset):
    """
    This defines a dataset, subclassed from torch.utils.data.Dataset.

    It uses pd.dataframe as the backend, with utility
    functions.

    Parameters:
            label_column: String
                The name of the label column.
                Provide None if you do not want a target.
            merge_on_columns: list of strings
                Key(s) that specifies which columns to use to uniquely stitch
                dataset (default None)
            index_list: list of strings
                List of columns to make the index of the dataframe
            na_values: The values to convert to NaN when reading from csv
            dataset_dict: keyword parameter, value is dataframe or path string
                pandas dataframe assigned to keyword argument that produces a
                dictionary variable.

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
        >>> tab_dataset_var = TabularDataset(merge_on_columns=['A'],
                            index_list=['A'], df1=df_test_one, df2=df_test_two)

    """

    def __init__(self, label_column=None, merge_on_columns=None,
                 index_list=None, na_values=None, **dataset_dict):
        """Creates an instance of TabularDataset."""
        for dataset_name in dataset_dict:
            dataset_value = dataset_dict[dataset_name]
            if isinstance(dataset_value, str):
                f_path = dataset_value
                dataset_dict[dataset_name] = pd.read_csv(f_path,
                                                         na_values=na_values,
                                                         index_col=index_list)
            elif not isinstance(dataset_value, pd.DataFrame):
                raise ValueError("Dataset inputs must be either paths \
                                 or DataFrame objects")

        if len(dataset_dict) == 1:
            # TODO: do we want to do anything with this name?
            self.df = dataset_dict[sorted(dataset_dict)[0]]
        else:
            # Not using index list now because we set it before
            self.df = utils.stitch_datasets(dataset_dict, merge_on_columns,
                                            index_list)
            # TODO: check index list doesn't fail if set twice...

        self.label_column = label_column

        self.df = utils.clean_dataframe(self.df)

    def __len__(self):
        """
        The total number of samples.

        Returns:
            The total number of samples: int
        """
        return len(self.df.index)

    def __getitem__(self, idx):
        """
        Generates one sample of data.

        Parameters:
            idx: int
                The index of the data

        Returns:
            The values of the row and the value of the label columns. Xs, Y

        """
        # Where df.drop is used to access the dataframe without
        # the label column, iloc gets the row, then access values and convert

        if self.label_column:
            if self.all_xs is None:
                self.all_xs = self.df.drop(self.label_column,
                              axis=1)
            if self.all_y is None:
                self.all_y = self.df[[self.label_column]]
            xs = torch.from_numpy(self.all_xs.iloc[[idx]].values[0]).float()
            y = self.all_y.iloc[[idx]].values.tolist()[0][0]
            y = torch.tensor(y, dtype=torch.long)
            return xs, y
        else:
            xs = self.df.iloc[[idx]].values.tolist()[0]
            xs = torch.tensor(xs, dtype=torch.float)
            return xs, None

    def merge_dataframe(self, merge_on_columns=None,
                        index_list=None, na_values=None, **dataset_dict):
        """
        Merges additional data into a TabularDataset isntance

        Parameters:
            merge_on_columns: list of strings
                Key(s) that specifies which columns to use to uniquely stitch
                dataset (default None)
            index_list: list of strings
                List of columns to make the index of the dataframe
            na_values: The values to convert to NaN when reading from csv
            dataset_dict: keyword parameter, value is dataframe or path string
                pandas dataframe assigned to keyword argument that produces a
                dictionary variable.

        """
        for dataset in dataset_dict:
            dict_value = dataset_dict[dataset]
            if isinstance(dict_value, str):
                f_path = dict_value
                dataset_dict[dataset] = pd.read_csv(f_path,
                                                    na_values=na_values,
                                                    index_col=index_list)
            elif not isinstance(dict_value, pd.DataFrame):
                raise ValueError("Dataset inputs must be either paths \
                                 or DataFrame objects")

        if len(dataset_dict) == 1:
            df = dataset_dict[sorted(dataset_dict)[0]]
            self.df = utils.stitch_datasets(df_main=self.df,
                                            merge_on_columns=merge_on_columns,
                                            index_list=index_list, new_df=df)
        else:
            if not self.df.empty:
                self.df = utils.stitch_datasets(df_main=self.df,
                                                merge_on_columns=
                                                merge_on_columns,
                                                index_list=index_list,
                                                **dataset_dict)
            else:
                self.df = utils.stitch_datasets(
                    merge_on_columns=merge_on_columns,
                    index_list=index_list,
                    **dataset_dict)

        logger.info("Successfully merged %d datasets", len(dataset_dict))

    def save_dataframe(self, file_path):
        """
        Save the dataframe to a file.

        Parameters:
            file_path: String
                Path to the file where you want your dataframe to be save

        """
        self.df.to_csv(file_path, encoding='utf-8', index=True)
        logger.info("You have saved the dataframe as a csv to %s", file_path)

    def list_all_features(self):
        """
        Lists all features (columns)
        """
        return self.df.columns.values.tolist()

    #  TODO: update to use kwargs.
    def replace_value_in_column(self, column_name, current_values,
                                target_values):
        """
        Replaces one or more values in the given columns.

        Parameters:
            column_name: String
            current_values: List
                Must be existing values
            target_values: List
                Must be valid for pandas dataframe

        """
        if not isinstance(current_values, list):
            current_values = [current_values]
            target_values = [target_values]

        if len(current_values) != len(target_values):
            raise ValueError(
                "Length of current values and target values must be the same")

        self.df[column_name] = self.df[column_name].replace(current_values,
                                                            target_values)

        logger.info("replaced values in %s", column_name)

    def list_all_column_values(self, column_name):
        """
        Return a list of all values in this column.

        Parameters:
            column_name: String
                Name of the column

        """
        return list(getattr(self.df, column_name).unique())

    def print_column_data_types(self):
        """Prints the data types of all columns. """
        print(self.df.dtypes)

    def delete_column(self, column_name):
        """
        Deletes the given column.

        Parameters:
            column_name: String
                The name of the column you want deleted

        """
        self.df = self.df.drop(column_name, axis=1)

        logger.info("You have dropped %s", column_name)

    # TODO: use kwargs
    def create_label_encoding(self, column_name, ordered_values):
        """
        Create label encoding for the given column

        Parameters:
            column_name: String
                The name of the column you want encoded
            ordered_values: List or Dict
                Either an ordered list of possible column values.
                Or a mapping of column value to label value.
                Must include all possible values

        """
        if isinstance(ordered_values, list):
            ordered_values = dict(map(lambda t: (t[1], t[0]), enumerate(
                ordered_values)))
        elif not isinstance(ordered_values, dict):
            raise ValueError("Must be either a list or dictionary")

        column_vals = ordered_values.keys()

        if len(column_vals) != len(set(column_vals)):
            raise ValueError("Ordered_value_list contains non-unique values")

        if set(column_vals) != set(self.list_all_column_values(column_name)):
            raise ValueError("Not all column values are included")

        self.df[column_name] = getattr(self.df, column_name).map(
            ordered_values)

        logger.info("Successfully remapped %s", column_name)

    def create_one_hot_encoding(self, column_name, prefix_sep="@"):
        """
        Create one-hot encoding for the given column.

        Parameters:
            column_name: String
                The name of the column you want to one-hot encode
            prefix_sep String default("@")
                The prefix used when creating a one-hot encoding

        """
        # TODO: ensure dummy_na =False is what you want
        self.df = pd.get_dummies(self.df, columns=[column_name],
                                    prefix_sep=prefix_sep)
        logger.info("Successfully encoded %s", column_name)

    # if a use case presents itself column_name could easily become a list
    def reverse_create_one_hot_encoding(self, column_name=None,
                                        prefix_sep="@"):
        """
        Undo the creation of one-hot encodings, if prefix_sep was used
        to create one-hot encodings and nowhere else. If a column_name is
        provided, only that column will be reverse-encoded, otherwise all will
        be reverse-encoded.

        Parameters
            column_name: String
                The name of the column to reverse the one-hot encoding for
            prefix_sep: String default("@")
                The prefix used when creating a one-hot encoding

        """
        result_series = {}

        if column_name:
            if prefix_sep in column_name:
                considered_column_list = [column_name]
            else:
                raise ValueError("Prefix not found in column_name")
            non_dummy_cols = [col for col in self.df.columns
                              if col not in considered_column_list]
        else:
            considered_column_list = self.df.columns
            non_dummy_cols = [col for col in self.df.columns
                              if prefix_sep not in col]

        # Find dummy columns and build pairs (category, category_value)
        dummy_tuples = [(col.split(prefix_sep)[0], col) for col in
                        considered_column_list if prefix_sep in col]

        for dummy, cols in groupby(dummy_tuples, lambda item: item[0]):
            dummy_df = self.df[[col[1] for col in cols]]

            # Find max value among columns
            max_columns = dummy_df.idxmax(axis=1)

            # Remove category_ prefix
            result_series[dummy] = max_columns.apply(lambda item:
                                                     item.split(prefix_sep)[1])

        # Copy non-dummy columns over.
        for col in non_dummy_cols:
            result_series[col] = self.df[col]

        self.df = pd.DataFrame(result_series)

        logger.info("Successfully converted %d \
                    columns back from dummy format.", len(dummy_tuples))

    def identify_null(self, threshold):
        """
        Return columns where there is at least threshold percent of
        null values.

        Parameters:
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
        num_threshold = ( (1- threshold) * len(self))
        # thresh is "Require that many non-NA values."
        tmp = self.df.dropna(thresh=num_threshold, axis=1)
        cols = list(set(self.df.columns).difference(set(tmp.columns)))
        return cols

    def identify_unique(self, threshold):
        """
        Returns columns that do not have have at least threshold number of
        values. If a column has 9 values and the threshold is 9, that column
        will not be returned.

        Parameters:
            threshold: The minimum number of values needed.
                Must be greater than 1. Not between 0 and 1, but rather
                represents the number of values

        Returns:
            column_list: list
                The list of columns having threshold number of values

        """
        obj_types = {col: set(map(type, self.df[col])) for col in
                     self.df.columns}

        column_list = []
        for col in self.df.columns:
            if len(obj_types[col]) > 1:
                logger.warning("Column: {} has mixed datatypes, this may"
                               "interfere with an accurate identification"
                               "of mixed values: i.e. you may have 1 and '1'"
                               .format(col))
            if len(self.df[col].unique()) < threshold:
                column_list.append(col)
        return column_list

    def identify_unbalanced_columns(self, threshold, non_numeric=True):
        """
        This returns columns that are highly unbalanced.
        Those that have a disproportionate amount of one value.

        Parameters:
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
            columns = list(self.df.columns)
        else:
            columns = list(self.df.select_dtypes(include=np.number))
        column_list = []
        for col in columns:
            # Check amount of null values, because if column is entirely null,
            # max won't work.
            num_of_null = self.df[col].isnull().sum()
            if num_of_null != len(self):
                col_maj = (max(self.df[col].value_counts()) /
                           self.df[col].value_counts().sum())
                if col_maj >= threshold:
                    column_list.append(col)
        return column_list

    def identify_highly_correlated(self, threshold):
        """
        Identify columns that are highly correlated with one-another.

        Parameters:
            threshold: Between 0 (weakest correlation) and 1
                (strongest correlation).
                Minimum amount of correlation necessary to be identified.

        Returns:
            column list: List of tuples
                The correlation values above threshold

        """
        column_list = set()
        features_correlation = self.df.corr().abs()
        corr_pairs = features_correlation.unstack()
        for index, val in corr_pairs.items():
            if val > threshold and (index[0] != index[1]):
                column_list.add((index, val))
        return column_list

    def identify_low_variance(self, threshold):
        """
        Identify those columns that have low variance

        Parameters:
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
        for col in self.df.columns:
            if self.df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                col_float_array = self.df[col].dropna().values.astype(float)
                reshaped_col_float_array = col_float_array.reshape(-1, 1)
                scaled_col = scaler.fit_transform(reshaped_col_float_array)
                col_var = scaled_col.var()
                if col_var <= threshold:
                    dct_low_var[col] = col_var
        return dct_low_var

    def convert_all_categorical_binary(self, list_only=False,
                                       exception_columns=None):
        """Recodes all columns with only two values as ones and zeros, float
        valued.

        This should only be used once you have a final dataset in case values
        are not actually binary. Useful for dealing with all the variations of
        YES/NO ,yEs/nO etc.

        Parameters:
             list_only: boolean
                only return a list of columns for which this would
                apply, do not actually do the transformation.
                If false operation is performed in place.
            exception_columns: list
                list of column names you do not wish to convert.

        Returns:
            list or None
            list if list_only if true, nothing otherwise.
        """

        # recoding binary valued columns as ones and zeros

        binary_cols = [(col, self.df[col].value_counts().index) for col
                       in self.df.columns if
                       len(self.df[col].value_counts()) == 2]

        if list_only:
            return binary_cols

        for col, index in binary_cols:
            di = {index[0]: 1.0, index[1]: 0.0}
            try:
                self.df = self.df.replace({col: di})
            except:
                continue
            self.df[col] = self.df[col].astype(np.float64)

    # future improvements could come from
    # https://github.com/pytorch/text/blob/master/torchtext/data/dataset.py
    # noinspection PyUnusedLocal
    def split(self, split_ratio=0.7, stratified=False, stratum_column=None):
        """
        Create train-test(-validation) splits from the instance's examples.
        Function signature borrowed from torchtext in an effort to maintain
        consistency
        https://github.com/pytorch/text/blob/master/torchtext/data/dataset.py
        also partially modified from
        https://stackoverflow.com/questions/38250710/
        how-to-split-data-into-3-sets-train-validation-and-test

        Parameters:
            split_ratio: Float
                a number [0, 1] denoting the amount
                of data to be used for the training split (rest is used for
                validation), or a list of numbers denoting the relative sizes
                of train, test and valid splits respectively.
                If the relative size for valid is missing, only the
                train-test split is returned.
                Default is 0.7 (for th train set).
            stratified: Boolean
                whether the sampling should be stratified.
                    Default is False.
            stratum_column: String
                name of the examples column stratified over.
                Default is 'label_column'

        Returns:
            datasets: Tuple of TabularDatasets
                Datasets for train, test, validation
                 splits in that order, if the splits are provided.

        """

        train_ratio, test_ratio, validation_ratio = utils.check_split_ratio(
            split_ratio)

        train_indices = []
        test_indices = []
        validation_indices = []

        if stratified:
            if not stratum_column:
                stratum_column = self.label_column
            else:
                if stratum_column not in self.df.columns:
                    raise ValueError("Invalid strata column name")

            grps = self.df.groupby(stratum_column)
            train_index, test_index, val_index = [], [], []
            for key, grp in grps:
                group_train, group_test, group_val = \
                    utils.rationed_split(grp, train_ratio, test_ratio,
                                         validation_ratio)

                train_indices += list(group_train)
                test_indices += list(group_test)
                validation_indices += list(group_val)

        else:

            train_indices, test_indices, validation_indices = \
                utils.rationed_split(self.df, train_ratio,
                                     test_ratio, validation_ratio)

        train = TabularDataset(train=self.df.loc[train_indices],
                               label_column=self.label_column)
        test = TabularDataset(test=self.df.loc[test_indices],
                              label_column=self.label_column)

        if validation_ratio:
            validation = TabularDataset(val=self.df.loc[validation_indices],
                                        label_column=self.label_column)
            return train, test, validation
        else:
            return train, test
