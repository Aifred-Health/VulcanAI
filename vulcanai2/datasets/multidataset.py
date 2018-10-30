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
        def get_total_targets(multi_datasets):
            num_targets = 0
            for ds in multi_datasets:
                if isinstance(ds, MultiDataset):
                    num_targets += get_total_targets(ds._datasets)
                else:
                    num_targets += int(ds[2])
            return num_targets
        self._datasets = datasets
        # must always have exactly one target.
        total_num_targets = get_total_targets(self._datasets)
        if total_num_targets > 1:
            raise ValueError(
                "You may specify at most one target."
                " {} specified".format(total_num_targets))

    def __len__(self):
        """
        Denotes the total number of samples.
        :return: None
        """
        logger.warning("Defaulting to the length of the smallest dataset")

        def get_min_length(multi_datasets):
            min_length = float('inf')
            for ds in multi_datasets:
                if isinstance(ds, MultiDataset):
                    length = get_min_length(ds._datasets)
                else:
                    length = len(ds[0])
                if length < min_length:
                    min_length = length
            return min_length

        return get_min_length(self._datasets)

    def __getitem__(self, idx):
        """
        Overrides getitem used by DataLoader required by torch Dataset.

        Parameters
        ----------
        idx : index
            Index of sample to extract

        Returns
        -------
        (input_data, targets) : (torch.Tensor, torch.Tensor)
            Tuple of input_data and target at that index as specific in config.
            Target data is always last.

        """
        input_data_items = []
        target_items = []

        for t in self._datasets:
            if isinstance(t, MultiDataset):
                # import pudb; pu.db

                input_data_items.append(t.__getitem__(idx)[0])
                try:
                    target_items.append(t.__getitem__(idx)[1])
                except IndexError:
                    # Targets don't exist
                    pass
            # Extract input data
            else:
                if t[1]:
                    ds = t[0]
                    # Assumes input_data is stored in the first slot of tuple.
                    input_data_items.append(ds.__getitem__(idx)[0])
                # Extract target data
                if t[2]:
                    ds = t[0]
                    # Assumes target is stored in the second slot of tuple.
                    target_items.append(ds.__getitem__(idx)[1])

        values = tuple([input_data_items] + target_items)
        return values
