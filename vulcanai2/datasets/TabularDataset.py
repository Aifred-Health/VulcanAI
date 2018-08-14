__author__ = 'Caitrin'
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from .utils import *

#TODO: this is copy pasted

class TabularDataset(Dataset):
    def __init__(self, file_path, headers=False, column_info=None):

        #TODO catch errors
        #TODO: add in header
        #TODO: need to convert to correct datatype

        if type(file_path) is list:

            self.data = self.merge_multiple_files() #TODO: instance method?

        #TODO: include a list of csv files and then you can merge them based on a certain column.... certain criteria must be assumed in advance
        #TODO: inspiration https://github.com/joemehltretter/aifred_ml/blob/master/COMED%20Prepoccesing.ipynb
        self.data = pd.read_csv(file_path)
        self.n = self.data.shape[0]

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
        return list(self.data.iloc[[idx]]), list(self.labels.iloc[[idx]])[0]


    def merge_multiple_files(self):
        pass

    def save_dataframe(self, file_path, include_labels=False):

        if include_labels:
            pass #TODO: append this

        self.data.to_csv(file_path, encoding='utf-8', index=True)


    #TODO: verify that labels has the same length as data
    #TODO: this will drop the label from the data
    def specify_label_column(self):
        pass

    def delete_columns(self, column_list):
        pass

    #TODO: this operates on categorical, as necessary. need to figure out where in the pipeline this happens...
    def create_dummies(self):
        pass

    def list_all_features(self):
        return list(self.data)

    def get_all_numeric_features(self):
        return [key for key in dict(self.data.dtypes) if dict(self.data.dtypes)[key] in ['float64', 'int64']]

    def get_all_categorical_features(self):
        return [key for key in dict(self.data.dtypes) if dict(self.data.dtypes)[key] not in ['float64', 'int64']]

    def remove_majority_null(self, threshold):

        pass #inspiration from https://github.com/joemehltretter/aifred_ml/blob/master/COMED%20Cleanup.ipynb

    def remove_unbalanced_columns(self, percentage_threshold, non_numeric=True):


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
        train_ratio, test_ratio, val_ratio = check_split_ratio(split_ratio)

        # For the permutations
        rnd = RandomShuffler(random_state)
        if not stratified:
            train_data, test_data, val_data = rationed_split(self.examples, train_ratio,
                                                             test_ratio, val_ratio, rnd)
        else:
            if strata_field not in self.fields:
                raise ValueError("Invalid field name for strata_field {}"
                                 .format(strata_field))
            strata = utils.stratify(self.examples, strata_field)
            train_data, test_data, val_data = [], [], []
            for group in strata:
                # Stratify each group and add together the indices.
                group_train, group_test, group_val = rationed_split(group, train_ratio,
                                                                    test_ratio, val_ratio,
                                                                    rnd)
                train_data += group_train
                test_data += group_test
                val_data += group_val

        splits = tuple(Dataset(d, self.fields)
                       for d in (train_data, val_data, test_data) if d)

        # In case the parent sort key isn't none
        if self.sort_key:
            for subset in splits:
                subset.sort_key = self.sort_key
        return splits