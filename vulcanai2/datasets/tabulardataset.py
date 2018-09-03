__author__ = 'Caitrin'
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from . import utils as utils


class TabularDataset(Dataset):
    def __init__(self, data, labelColumn="label", joinColumn=None, indexList=None):
        """
        Creates an instance of Tabulardataset

        :param data: Either a path to a csv file, a list of paths to csv files or a dataframe
        :param labelColumn: Default label; the name of the column used as the y or label value
        :param joinColumn: The column on which a list of datasets should be joined
        :param index_list:
        :return:
        """
        if isinstance(data, list):
            if not joinColumn:
                raise RuntimeError("You need to provide a joinColumn if a list of csvs are provided")
            dfs = [pd.read_csv(f) for f in data]
            self.df = utils.stitch_datasets(dfs, joinColumn, indexList)
        elif isinstance(data, pd.DataFrame):
            self.df = data
        else:
            self.df = pd.read_csv(data)

        self.labelColumn = labelColumn
        self.n = self.df.shape([0]) #TODO: delete this cause it won't update and that's dumb cause you're letting people access the dataframe object

    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return self.n

    def __getitem__(self, idx):
        """
        Generates one sample of data
        """
        #TODO: check types, possibly wrong
        return list(self.df.drop(self.labelColumn, axis=1)[[idx]]), list(self.df[[self.labelColumn]].iloc[[idx]])[0]

    def save_dataframe(self, file_path):
        """
        Save the dataframe to a file.
        :param file_path: the file path
        :return: Noneff
        """
        self.data.to_csv(file_path, encoding='utf-8', index=True)

    #TODO: this is kinda useless
    def delete_columns(self, columnList):
        self.df = self.df.drop([columnList])

    #TODO: this operates on categorical, as necessary. need to figure out where in the pipeline this happens...
    def create_dummies(self):
        pass

    def list_all_features(self):
        """
        lists all features
        :return: returns a list of all features.
        """
        return list(self.df)

    def get_all_numeric_features(self):
        """
        Returns all columns that contain numeric values
        :return: all columns that contain numeric values
        """
        return [key for key in dict(self.df.dtypes) if dict(self.df.dtypes)[key] in ['float64', 'int64']]

    def get_all_categorical_features(self):
        """
        Returns all columns that contain categorical values
        :return: all columns that contain categorical values
        """
        return [key for key in dict(self.df.dtypes) if dict(self.df.dtypes)[key] not in ['float64', 'int64']]

    #TODO: check this doesn't operate in place... damn
    def remove_majority_null(self, threshold):
        """
        Remove columns where the majority of values as determined by the threshold are numm
        :param threshold: A number between 0 and 1
        :return:
        """
        if threshold >= 1 or threshold <= 0:
            raise ValueError("Threshold needs to be a proportion between 0 and 1")
        num_threshold = threshold* self.n
        self. df = self.df.dropna(thresh = num_threshold, axis=1)

    def remove_unique(self, threshold):
        """
        Removes columns that have less than threshold number of unique values
        :param threshold: All columns that have threshold or less unique values will be removed.
        :return:
        """
        for col in self.df.columns:
            if len(self.df[col].unique()) <= threshold:
                self.df = self.df.drop(col,axis=1)

    #TODO: implement variance thresholding
    def remove_unbalanced_columns(self, threshold, non_numeric=True):
        raise NotImplementedError

    def remove_highly_correlated(self, threshold, non_numberic=True):
        raise NotImplementedError

    #TODO: edit this method that creates a split given different filepaths or objects so that the params match
    #taken from https://github.com/pytorch/text/blob/master/torchtext/data/dataset.py
    # @classmethod
    # def splits(cls, path=None, root='.data', train=None, validation=None,
    #            test=None, **kwargs):
    #     """Create Dataset objects for multiple splits of a dataset.
    #     Arguments:
    #         path (str): Common prefix of the splits' file paths, or None to use
    #             the result of cls.download(root).
    #         root (str): Root dataset storage directory. Default is '.data'.
    #         train (str): Suffix to add to path for the train set, or None for no
    #             train set. Default is None.
    #         validation (str): Suffix to add to path for the validation set, or None
    #             for no validation set. Default is None.
    #         test (str): Suffix to add to path for the test set, or None for no test
    #             set. Default is None.
    #         Remaining keyword arguments: Passed to the constructor of the
    #             Dataset (sub)class being used.
    #     Returns:
    #         Tuple[Dataset]: Datasets for train, validation, and
    #             test splits in that order, if provided.
    #     """
    #     if path is None:
    #         path = cls.download(root)
    #     train_data = None if train is None else cls(
    #         os.path.join(path, train), **kwargs)
    #     val_data = None if validation is None else cls(
    #         os.path.join(path, validation), **kwargs)
    #     test_data = None if test is None else cls(
    #         os.path.join(path, test), **kwargs)
    #     return tuple(d for d in (train_data, val_data, test_data)
    #                  if d is not None)


    #Function signature borrowed from torchtext in an effort to maintain consistency
    #https://github.com/pytorch/text/blob/master/torchtext/data/dataset.py
    #also partially modified from https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test
    def split(self, split_ratio=0.7, stratified=False, strata_field='label',
              random_state=None):
        """Create train-test(-valid?) splits from the instance's examples.
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
            raise NotImplementedError("We need to do this!")

        train_ratio, test_ratio, val_ratio = utils.check_split_ratio(split_ratio)

        np.random.seed(random_state)
        perm = np.random.permutation(self.df.index)
        m = len(self.df.index)

        train_end = int(train_ratio * m)
        train_df = self.df.loc[perm[:train_end]]
        if val_ratio:
            val_end = int(val_ratio * m) + train_end
            val_df = self.df.loc[perm[train_end:val_end]] #todo: triple verify its loc and not iloc
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
