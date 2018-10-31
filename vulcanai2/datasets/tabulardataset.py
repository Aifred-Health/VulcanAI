# -*- coding: utf-8 -*-
"""
This file defines the TabularDataset Class
"""
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from . import utils as utils
import logging
from itertools import groupby
from sklearn import preprocessing

logger = logging.getLogger(__name__)


class TabularDataset(Dataset):
    """
    This defines a dataset, subclassed from torch.utils.data.Dataset. It uses pd.dataframe as the backend, with utility
    functions.
    """
    def __init__(self, data, label_column=None, join_column=None, index_list=None):
        """
        Creates an instance of Tabulardataset
        :param data: Either a path to a csv file, a list of paths to csv files or a dataframe
        :param label_column: Default label; the name of the column used as the y or label value
        :param join_column: The column on which a list of datasets should be joined
        :param index_list: List of feature columns to add
        :return: None
        """
        if isinstance(data, list):
            if not join_column:
                raise RuntimeError("You need to provide a join_column if a list of csvs are provided")
            dfs = [pd.read_csv(f) for f in data]
            self.df = utils.stitch_datasets(dfs, join_column, index_list)
        elif isinstance(data, pd.DataFrame):
            self.df = data
        else:
            self.df = pd.read_csv(data)
        self.labelColumn = label_column

        self.df = utils.clean_dataframe(self.df)

        dataset_length = self.__len__()
        logger.info(f"You have created a new dataset with {dataset_length} rows")

    def __len__(self):
        """
        Denotes the total number of samples.
        :return: None
        """
        return self.df.shape[0]

    def __getitem__(self, idx):
        """
        Generates one sample of data
        :param idx: The index of the data
        :return: The values of the row and the value of the label columns. Xs and y.
        """
        # Where df.drop is used to access the dataframe without the label column, iloc gets the row, then access values
        # and convert
        if self.labelColumn:
            xs = self.df.drop(self.labelColumn, axis=1).iloc[[2]].values.tolist()[0]
            y = self.df[[self.labelColumn]].iloc[[idx]].values.tolist()[0]
            return xs, y
        else:
            xs = self.df.iloc[[2]].values.tolist()[0]
            return xs

    def convert_to_dataframe(self):
        """
        Converts TabularDataset variable back to dataframe
        :return: The dataframe at current state
        """
        return self.df

    def save_dataframe(self, file_path):
        """
        Save the dataframe to a file.
        :param file_path: the file path
        :return: Noneff
        """
        self.df.to_csv(file_path, encoding='utf-8', index=True)
        logger.info(f"You have saved the dataframe as a csv to {file_path}")

    def list_all_features(self):
        """
        Lists all features (columns)
        :return: returns a list of all features.
        """
        return list(self.df)

    def replace_value_in_column(self, columns, current_values, target_values):
        """
        Replace one or more values in either a single column or a list of columns.
        :param columns: Either a single column or list of columns where you want the values to be replaced
        :param current_values: Either a single value or a list of values that you want to be replaced
        :param target_values: Either a single value or a list of values you want the current value to be replaced with
        :return: None
        """

        if not isinstance(columns, list):
            columns = [columns]

        if not isinstance(current_values, list):
            current_values = [current_values]
            target_values = [target_values]

        if len(current_values) != len(target_values):
            raise ValueError("Length of current values and target values must be the same")

        self.df[columns] = self.df[columns].replace(current_values, target_values)

        logger.info(f"replaced values in {columns}")

    def list_all_column_values(self, column_name):
        """
        List all values in this column
        :param column_name:
        :return:
        """
        return list(getattr(self.df, column_name)().unique())

    # TODO: this is really slow make it faster
    def identify_all_numerical_features(self):
        """
        Returns all columns that contain numeric values
        :return: all columns that contain numeric values
        """
        return [key for key in dict(self.df.dtypes)
                if dict(self.df.dtypes)[key] in ['float64', 'int64', 'float32', 'int32']]

    # TODO: this is really slow make it faster
    def identify_all_categorical_features(self):
        """
        Returns all columns that contain categorical values
        :return: all columns that contain categorical values
        """
        return [key for key in dict(self.df.dtypes)
                if dict(self.df.dtypes)[key] not in ['float64', 'int64', 'float32', 'int32']]

    def delete_columns(self, column_list):
        """
        Deletes columns in the list
        :param column_list: List of columns to be deleted
        :return: None
        """
        prior_length = self.__len__()
        for col in column_list:
            if col in list(self.df):
                self.df = self.df.drop(col, axis=1)

        cur_length = self.__len__()
        logger.info(f"You have dropped {prior_length - cur_length} columns")

    def create_label_encoding(self, column, ordered_values):
        """
        Create label encoding for
        Used for those categorical features where order does matter.
        :param ordered_values: Either an ordered list of possible column values. Or a mapping of column value to \
        label value. Must include all possible values.
        :return:
        """

        if isinstance(ordered_values, list):
            ordered_values = dict(map(lambda t: (t[1], t[0]), enumerate(ordered_values)))
        elif not isinstance(ordered_values, dict):
            raise ValueError("Must be either a list or dictionary")

        column_vals = ordered_values.keys()

        if len(column_vals) != len(set(column_vals)):
            raise ValueError("Ordered_value_list contains non-unique values")

        if set(column_vals) != self.list_all_column_values(column):
            raise ValueError("Not all column values are included")

        self.df = getattr(self.df, column)().map(ordered_values)

        logger.info(f"Successfully remapped {column}")

    def create_one_hot_encoding(self, column, prefix_sep="@"):
        """
        Create one-hot encoding for the provided column. Deliberatly doesn't make any decisions for you.
        :param column: The name of the column you want to one-hot encode
        :param prefix_sep: The prefix seperator.
        :return: None
        """
        if column in list(self.df):
                self.df = pd.get_dummies(self.df, dummy_na=True, columns=[column], prefix_sep=prefix_sep)
        else:
                logger.info(f"Col {column} does not exist")

    def reverse_create_one_hot_encoding(self, prefix_sep="@"):
        """
        Ensure prefix sep only exists for dummy columns
        :return: None
        """
        result_series = {}

        # Find dummy columns and build pairs (category, category_value)
        dummy_tuples = [(col.split(prefix_sep)[0], col) for col in self.df.columns if prefix_sep in col]

        # Find non-dummy columns that do not have a _
        non_dummy_cols = [col for col in self.df.columns if prefix_sep not in col]

        # For each category column group use idxmax to find the value.
        for dummy, cols in groupby(dummy_tuples, lambda item: item[0]):
            # Select columns for each category
            dummy_df = self.df[[col[1] for col in cols]]

            # Find max value among columns
            max_columns = dummy_df.idxmax(axis=1)

            # Remove category_ prefix
            result_series[dummy] = max_columns.apply(lambda item: item.split(prefix_sep)[1])

        # Copy non-dummy columns over.
        for col in non_dummy_cols:
            result_series[col] = self.df[col]

        self.df = pd.DataFrame(result_series)

        logger.info(f"Successfully converted {len(dummy_tuples)} columns back from dummy format.")

    def identify_majority_null(self, threshold):
        """
        Return columns where the number of values as determined by the threshold are null
        :param threshold: A number between 0 and 1, representing proportion needed to drop
        :return: A list of those columns with threshold percentage null values
        """
        if threshold >= 1 or threshold <= 0:
            raise ValueError("Threshold needs to be a proportion between 0 and 1")
        num_threshold = threshold * self.__len__()
        tmp = self.df.dropna(thresh=num_threshold, axis=1)
        cols = list(set(tmp.columns).difference(set(self.df.columns)))
        return cols

    def identify_unique(self, threshold):
        """
        Returns columns that have less than threshold number of unique values
        :param threshold: All columns that have threshold or less unique values will be removed.
        :return: The column list
        """
        column_list = []
        for col in self.df.columns:
            if len(self.df[col].unique()) <= threshold:
                column_list.append(col)
        return column_list

    # TODO: add in non_numeric
    def identify_unbalanced_columns(self, threshold, non_numeric=True):
        """
        This returns columns that are highly unbalanced, aka those that have a disproportionate amount of one value
        :param threshold: Proportion needed to define unbalanced, between 0 and 1
        :param non_numeric: Whether non-numeric columns are also considered.
        :return: The column list
        """
        if non_numeric:
            columns = list(self.df.columns)
        else:
            columns = list(self.df.select_dtypes(include=np.number))
        column_list = []
        for col in columns:
            #Check amount of null values, because if column is entirely null, max won't work.
            num_of_null = self.df[col].isnull().sum()
            if num_of_null != self.__len__():
                col_maj = (max(self.df[col].value_counts()) / self.df[col].value_counts().sum())
                if col_maj <= threshold:
                    column_list.append(col)
        return column_list

    def identify_highly_correlated(self, threshold):
        """
        Remove one of those columns that are highly correlated with one-another.
        :param threshold: Amount of correlation necessary for removal.
        :return: None
        """
        raise NotImplementedError

    def identify_low_variance(self, threshold):
        """
        Removes those columns that have low variance
        :param threshold: Upper bound of variance needed for removal
        :return: A dictionary of column names, with the value being their variance
        """
        dct_low_var = {}
        scaler = preprocessing.MinMaxScaler()
        for col in self.df.columns:
            if self.df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                col_float_array = self.df[[col]].values.astype(float)
                scaled_col = scaler.fit_transform(col_float_array)
                col_var = scaled_col.var()
                if col_var <= threshold:
                    dct_low_var[col] = col_var
        return dct_low_var

    # future improvements could come from https://github.com/pytorch/text/blob/master/torchtext/data/dataset.py
    # noinspection PyUnusedLocal
    def split(self, split_ratio=0.7, stratified=False, strata_field='label',
              random_state=None):
        """Create train-test(-valid?) splits from the instance's examples.
        Function signature borrowed from torchtext in an effort to maintain consistency
        https://github.com/pytorch/text/blob/master/torchtext/data/dataset.py
        also partially modified from
        https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test
        Arguments:
            split_ratio (float or List of floats): a number [0, 1] denoting the amount
                of data to be used for the training split (rest is used for validation),
                or a list of numbers denoting the relative sizes of train, test and valid
                splits respectively. If the relative size for valid is missing, only the
                train-test split is returned. Default is 0.7 (for th train set).
            stratified (bool): whether the sampling should be stratified.
                Default is False.
            strata_field (str): name of the examples Field stratified over.
                Default is 'label' for the conventional label field.
            random_state (int): the random seed used for shuffling.
        Returns:
            Tuple[Dataset]: Datasets for train, validation, and
                test splits in that order, if the splits are provided.
        """

        if stratified:
            raise NotImplementedError("We still need to get to this!")

        train_ratio, test_ratio, val_ratio = utils.check_split_ratio(split_ratio)

        print(train_ratio, test_ratio, val_ratio)

        np.random.seed(random_state)
        perm = np.random.permutation(self.df.index)
        m = len(self.df.index)

        train_end = int(train_ratio * m)
        train_df = self.df.loc[perm[:train_end]]
        val_df = None  # just to shut up linter
        if val_ratio:
            val_end = int(val_ratio * m) + train_end
            val_df = self.df.loc[perm[train_end:val_end]]
            test_start = val_end
        else:
            test_start = train_end
        test_df = self.df.loc[perm[test_start:]]

        train = TabularDataset(train_df, self.labelColumn)
        test = TabularDataset(test_df, self.labelColumn)
        if val_ratio:
            val = TabularDataset(val_df, self.labelColumn)
            return train, val, test
        else:
            return train, test
