# -*- coding: utf-8 -*-
"""
This file defines the TabularDataset Class
"""
import torch
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
    This defines a dataset, subclassed from torch.utils.data.Dataset.
    It uses pd.dataframe as the backend, with utility
    functions.
    """
    def __init__(self, label_column=None, merge_on_columns=None,
                 index_list=None, na_values=None, **kwargs):
        """
        Creates an instance of TabularDataset

        Parameters
        ----------
        label_column: String
            The name of the label column.
            Provide None if you do not want a target.
        merge_on_columns: list of strings
            Key(s) that specifies which columns to use to uniquely stitch
            dataset (default None)
        index_list: list of strings
            List of columns to make the index of the dataframe
        na_values: The values to convert to NaN when reading from csv
        kwargs: keyword parameter, value is either path or dataframe
            Where key: dataset name and value is either a path to a file
            or a dataframe.
        """
        dataset_dict = kwargs
        for dataset in dataset_dict:
            v = dataset_dict[dataset_dict]
            if isinstance(v, str):
                f_path = v
                dataset_dict[dataset] = pd.read_csv(f_path,
                                                    na_values=na_values,
                                                    index_col=index_list)
            elif not isinstance(v, pd.DataFrame):
                raise ValueError("Dataset inputs must be either paths \
                                 or DataFrame objects")

        if len(dataset_dict) == 1:
            key, value = sorted(dataset_dict)[0]  # anyone got smthing better?
            self.df = value  # TODO: do we want to do anything with this name?
        else:
            # Not using index list now because we set it before
            self.df = utils.stitch_datasets(dataset_dict, merge_on_columns,
                                            index_list)
            # TODO: check index list doesn't fail if set twice...

        self.label_column = label_column

        self.df = utils.clean_dataframe(self.df)

        logger.info(f"You have created a new dataset with {len(self)} rows")

    def __len__(self):
        """
        The total number of samples

        Returns
        -------
        The total number of samples: int
        """
        return self.df.shape[0]

    def __getitem__(self, idx):
        """
        Generates one sample of data

        Parameters
        ----------
        idx: int
            The index of the data

        Returns
        -------
        The values of the row and the value of the label columns. Xs, Y
        """

        # Where df.drop is used to access the dataframe without
        # the label column, iloc gets the row, then access values and convert

        if self.label_column:
            xs = self.df.drop(self.label_column,
                              axis=1).iloc[[2]].values.tolist()[0]
            xs = torch.tensor(xs, dtype=torch.float)
            y = self.df[[self.label_column]].iloc[[idx]].values.tolist()[0][0]
            y = torch.tensor(y, dtype=torch.long)
            return xs, y
        else:
            xs = self.df.iloc[[2]].values.tolist()[0]
            print(xs)
            xs = torch.tensor(xs, dtype=torch.float)
            return xs

    def merge_data(self, merge_on_columns=None,
                   index_list=None, na_values=None, **kwargs):
        """
        Merges additional data into a TabularDataset isntance

        Parameters
        ----------
        merge_on_columns: list of strings
            Key(s) that specifies which columns to use to uniquely stitch
            dataset (default None)
        index_list: list of strings
            List of columns to make the index of the dataframe
        na_values: The values to convert to NaN when reading from csv
        kwargs: keyword parameter, value is either path or dataframe
            Where key: dataset name and value is either a path to a file
            or a dataframe.
        """

        dataset_dict = kwargs

        if "original" in dataset_dict:
            raise ValueError("Please choose a name other than 'original'")

        dataset_dict["original"] = self.df

        for dataset in dataset_dict:
            v = dataset_dict[dataset_dict]
            if isinstance(v, str):
                f_path = v
                dataset_dict[dataset] = pd.read_csv(f_path,
                                                    na_values=na_values,
                                                    index_col=index_list)
            elif not isinstance(v, pd.DataFrame):
                raise ValueError("Dataset inputs must be either paths \
                                 or DataFrame objects")

        if len(dataset_dict) == 1:
            key, value = sorted(dataset_dict)[0]  # anyone got smthing better?
            self.df = value  # TODO: do we want to do anything with this name?
        else:
            # Not using index list now because we set it before
            self.df = utils.stitch_datasets(dataset_dict, merge_on_columns,
                                            index_list)

        logger.info(f"Successfully merged {len(kwargs)} datasets")

    def save_dataframe(self, file_path):
        """
        Save the dataframe to a file.

        Parameters
        ----------
        file_path: String
            Path to the file where you want your dataframe to be saved
        """
        self.df.to_csv(file_path, encoding='utf-8', index=True)
        logger.info(f"You have saved the dataframe as a csv to {file_path}")

    def list_all_features(self):
        """
        Lists all features (columns)
        :return: returns a list of all features.
        """
        return self.df.columns.values.tolist()

    def replace_value_in_column(self, column_name, current_values,
                                target_values):
        """
        Replace one or more values in either a single column or a list of columns.
        :param column: The column where you want values to be replaced
        :param current_values: Either a single value or a list of values that you want to be replaced
        :param target_values: Either a single value or a list of values you want the current values to be replaced with
        :return: None
        """

        if not isinstance(current_values, list):
            current_values = [current_values]
            target_values = [target_values]

        if len(current_values) != len(target_values):
            raise ValueError("Length of current values and target values must be the same")

        self.df[column_name] = self.df[column_name].replace(current_values, target_values)

        logger.info(f"replaced values in {column_name}")

    def list_all_column_values(self, column_name):
        """
        Return a list of all values in this column
        :param column_name:
        :return: All unique values in a column.
        """
        return list(getattr(self.df, column_name).unique())

    def print_column_data_types(self):
        """
        Prints the data types of all columns
        Returns None
        -------
        """
        print(self.df.dtypes)

    def delete_column(self, column_name):
        """
        Deletes the given column
        :param column_list: The name of the column to be deleted.
        :return: None
        """
        if column_name in list(self.df):
            self.df = self.df.drop(column_name, axis=1)

        logger.info(f"You have dropped {column_name}")

    def create_label_encoding(self, column_name, ordered_values):
        """
        Create label encoding for the provided column.
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

        if set(column_vals) != set(self.list_all_column_values(column_name)):
            raise ValueError("Not all column values are included")

        self.df[column_name] = getattr(self.df, column_name).map(ordered_values)

        logger.info(f"Successfully remapped {column_name}")

    def create_one_hot_encoding(self, column_name, prefix_sep="@"):
        """
        Create one-hot encoding for the provided column. Deliberatly doesn't make any decisions for you.
        :param column: The name of the column you want to one-hot encode
        :param prefix_sep: The prefix seperator.
        :return: None
        """
        #TODO: ensure dummy_na =False is what you want
        if column_name in list(self.df):
            self.df = pd.get_dummies(self.df, dummy_na=False, columns=[column_name], prefix_sep=prefix_sep)
            logger.info(f"Successfully encoded {column_name}")

        else:
            logger.info(f"Col {column_name} does not exist")

    def reverse_create_all_one_hot_encodings(self, prefix_sep="@"):
        """
        Ensure prefix sep only exists in dummy columns
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

    def identify_sufficient_non_null(self, threshold):
        """
        Return columns where there is at least threshold percent of non-null values.
        Does not drop columns
        :param threshold: A number between 0 and 1, representing proportion needed
        :return: A list of those columns with threshold percentage null values
        """
        if threshold >= 1 or threshold <= 0:
            raise ValueError("Threshold needs to be a proportion between 0 and 1")
        num_threshold = (threshold * len(self))
        tmp = self.df.dropna(thresh=num_threshold, axis=1)  # thresh is "Require that many non-NA values."
        cols = list(set(self.df.columns).difference(set(tmp.columns)))
        return cols

    def identify_unique(self, threshold):
        """
        Returns columns that have at least threshold number of values
        :param threshold: All columns having at least threshold number of unique values.
        """
        column_list = []
        for col in self.df.columns:
            if len(self.df[col].unique()) >= threshold:
                column_list.append(col)
        return column_list

    def identify_unbalanced_columns(self, threshold, non_numeric=True):
        """
        This returns columns that are highly unbalanced, aka those that have a disproportionate amount of one value.
        Return those columns that have at least threshold percentage of one value.
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
            if num_of_null != len(self):
                col_maj = (max(self.df[col].value_counts()) / self.df[col].value_counts().sum())
                if col_maj >= threshold:
                    column_list.append(col)
        return column_list

    def identify_highly_correlated(self, threshold):
        """
        Identify columns that are highly correlated with one-another.
        :param threshold: Between 0 and 1. Minimum amount of correlation necessary to be identified.
        :return: None
        """
        column_list = set()
        featuresCorrelation = self.df.corr().abs()
        corr_pairs = featuresCorrelation.unstack()
        for index, val in corr_pairs.items():
            if val > threshold and (index[0] != index[1]):
                column_list.add((index,val))
        return column_list

    # TODO: is it ok that this is maxiumum amount of variance?
    def identify_low_variance(self, threshold):
        """
        Identify those columns that have low variance
        :param threshold: Between 0 an 1. Maximum amount of variance necessary to be identified as low variance.
        :return: A dictionary of column names, with the value being their variance
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

    # TODO: re-write documentation
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

        train = TabularDataset(train_df, self.label_column)
        test = TabularDataset(test_df, self.label_column)
        if val_ratio:
            val = TabularDataset(val_df, self.label_column)
            return train, val, test
        else:
            return train, test
