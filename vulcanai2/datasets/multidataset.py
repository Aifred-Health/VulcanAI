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

    Parameters
    ----------
    datasets : list of tuple(Dataset, use_data_boolean, use_target_boolean)
        A list of tuples, wherein each tuple should have the Dataset in the
        zero index, a boolean of whether or not to include the input_data in
        the first index, and a boolean of whether or not to include the target
        data in the second index. You can only specificy one target at a time
        Throughout all incoming datasets.

    Returns
    -------
    multi_dataset : torch.utils.data.Dataset

    """

    # TODO: is datasets a reasonable name?
    # TODO: would be better to make these namedtuples...
    def __init__(self, datasets):
        """Initialize a dataset for multi input networks."""
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

        Will look for the dataset with the smallest number of samples and
        default the length to that so as to avoid getting a sample that doesn't
        exist in another dataset.

        Returns
        -------
        length : int

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
        Override getitem used by DataLoader required by torch Dataset.

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
        target_item = None

        for t in self._datasets:
            if isinstance(t, MultiDataset):
                # import pudb; pu.db
                input_data_items.append(t.__getitem__(idx)[0])
                try:
                    target_item = t.__getitem__(idx)[1]
                except IndexError:
                    # Targets don't exist
                    pass
            # Extract input data
            else:
                if t[1]: #TODO: rename these
                    ds = t[0]
                    # Assumes input_data is stored in the first slot of tuple.
                    input_data_items.append(ds.__getitem__(idx)[0])
                # Extract target data
                if t[2]:
                    ds = t[0]
                    # Assumes target is stored in the second slot of tuple.
                    target_item = ds.__getitem__(idx)[1] #technically would re-write if they had 2 targets...

        values = input_data_items, target_item
        return values
