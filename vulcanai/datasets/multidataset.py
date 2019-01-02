# coding=utf-8
""" Defines the MultiDataset Class"""
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class MultiDataset(Dataset):
    """
    Define a dataset for multi input networks.

    Takes in a list of datasets, and whether or not their input_data
    and target data should be output.

    Parameters:
        dataset_tuples : list of tuples
            Each tuple being (Dataset, use_data_boolean, use_target_boolean).
            A list of tuples, wherein each tuple should have the Dataset in the
            zero index, a boolean of whether to include the input_data in
            the first index, and a boolean of whether to include the target
            data in the second index. You can only specificy one target at a
            time throughout all incoming datasets.

    Returns:
        multi_dataset : torch.utils.data.Dataset

    """

    def __init__(self, dataset_tuples):
        """Initialize a dataset for multi input networks."""
        def get_total_targets(multi_datasets):
            num_targets = 0
            for tup in multi_datasets:
                if isinstance(tup, MultiDataset):
                    num_targets += get_total_targets(tup._dataset_tuples)
                else:
                    num_targets += int(tup[2])
            return num_targets

        self._dataset_tuples = dataset_tuples
        # must always have exactly one target.
        total_num_targets = get_total_targets(self._dataset_tuples)
        if total_num_targets > 1:
            raise ValueError(
                "You may specify at most one target."
                " {} specified".format(total_num_targets))

    def __len__(self):
        """
        Denotes the total number of samples.

        Will look for the dataset with the smallest number of samples and
        default the length to that so as to avoid getting a sample that doesn't
        exist in another dataset.

        Returns:
            length : int

        """
        logger.warning("Defaulting to the length of the smallest dataset")

        def get_min_length(multi_datasets):
            min_length = float('inf')
            for tup in multi_datasets:
                if isinstance(tup, MultiDataset):
                    length = get_min_length(tup._dataset_tuples)
                else:
                    length = len(tup[0])
                if length < min_length:
                    min_length = length
            return min_length

        return get_min_length(self._dataset_tuples)

    def __getitem__(self, idx):
        """
        Override getitem used by DataLoader required by torch Dataset.

        Parameters:
            idx : index
                Index of sample to extract

        Returns:
            (input_data, targets) : (torch.Tensor, torch.Tensor)
                Tuple of input_data and target at that index as specific in
                config. Target data is always last.

        """
        input_data_items = []
        target_item = None

        for tup in self._dataset_tuples:

            include_data = tup[1]
            include_target = tup[2]

            if isinstance(tup, MultiDataset):
                ds = tup
            else:
                ds = tup[0]

            if include_data:
                input_data_items.append(ds.__getitem__(idx)[0])

            if include_target:
                target_item = ds.__getitem__(idx)[1]

        return input_data_items, target_item
