# coding=utf-8
""" Defines the MultiDataset Class"""
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class MultiDataset(Dataset):
    """
    Defines a dataset for multi input networks.
    """
    # TODO: is datasets a reasonable name?
    # TODO: would be better to make these namedtuples...
    def __init__(self, datasets):
        """
        Takes in a list of datasets, and whether or not their input_data and target data should be output
        :param datasets: A list of tuples, where each tuple is in the form. Can only ever specificy one target.
        (Dataset Object, input_data_boolean, target_data_boolean)
        """

        self._datasets = datasets

        # must always have exactly one target.
        if sum([x[2] for x in self._datasets]) != 1:
            raise ValueError("You may specify only one target")

    def __len__(self):
        """
        Denotes the total number of samples.
        :return: None
        """
        logger.warning("Defaulting to the length of the smallest dataset")
        return min([t[0].__len__() for t in self._datasets])

    def __getitem__(self, idx):
        """
        Defines get item used by DataLoader as required by torch.utils.data.Dataset
        :param idx: index
        :return: a tuple of input_data and target at that index as specific in config. target data is always last.
        """
        input_data_items = []
        target_items = []

        for t in self._datasets:
            if t[1]:
                ds = t[0]
                input_data_items.append(ds.__getitem__(idx)[0])  # this assumes input_data is returned first

        for t in self._datasets:
            if t[2]:
                ds = t[0]
                target_items.append(ds.__getitem__(idx)[0])

        vals = tuple(input_data_items + target_items)
        return vals
