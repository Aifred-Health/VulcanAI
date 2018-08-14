__author__ = 'Caitrin'

import random
from contextlib import contextmanager
from copy import deepcopy

#TODO: this was taken from pytorch code.... but needs to be adapted to work with pytorch data
#https://github.com/pytorch/text/blob/master/torchtext/data/dataset.py

def check_split_ratio(split_ratio):
    """Check that the split ratio argument is not malformed"""
    valid_ratio = 0.
    if isinstance(split_ratio, float):
        # Only the train set relative ratio is provided
        # Assert in bounds, validation size is zero
        assert split_ratio > 0. and split_ratio < 1., (
            "Split ratio {} not between 0 and 1".format(split_ratio))

        test_ratio = 1. - split_ratio
        return (split_ratio, test_ratio, valid_ratio)
    elif isinstance(split_ratio, list):
        # A list of relative ratios is provided
        length = len(split_ratio)
        assert length == 2 or length == 3, (
            "Length of split ratio list should be 2 or 3, got {}".format(split_ratio))

        # Normalize if necessary
        ratio_sum = sum(split_ratio)
        if not ratio_sum == 1.:
            split_ratio = [float(ratio) / ratio_sum for ratio in split_ratio]

        if length == 2:
            return tuple(split_ratio + [valid_ratio])
        return tuple(split_ratio)
    else:
        raise ValueError('Split ratio must be float or a list, got {}'
                         .format(type(split_ratio)))


def stratify(examples, strata_field):
    # The field has to be hashable otherwise this doesn't work
    # There's two iterations over the whole dataset here, which can be
    # reduced to just one if a dedicated method for stratified splitting is used
    unique_strata = set(getattr(example, strata_field) for example in examples)
    strata_maps = {s: [] for s in unique_strata}
    for example in examples:
        strata_maps[getattr(example, strata_field)].append(example)
    return list(strata_maps.values())


def rationed_split(examples, train_ratio, test_ratio, val_ratio, rnd):
    # Create a random permutation of examples, then split them
    # by ratio x length slices for each of the train/test/dev? splits
    N = len(examples)
    randperm = rnd(range(N))
    train_len = int(round(train_ratio * N))

    # Due to possible rounding problems
    if not val_ratio:
        test_len = N - train_len
    else:
        test_len = int(round(test_ratio * N))

    indices = (randperm[:train_len],  # Train
               randperm[train_len:train_len + test_len],  # Test
               randperm[train_len + test_len:])  # Validation

    # There's a possibly empty list for the validation set
    data = tuple([examples[i] for i in index] for index in indices)

    return data




class RandomShuffler(object):
    """Use random functions while keeping track of the random state to make it
    reproducible and deterministic."""

    def __init__(self, random_state=None):
        self._random_state = random_state
        if self._random_state is None:
            self._random_state = random.getstate()

    @contextmanager
    def use_internal_state(self):
        """Use a specific RNG state."""
        old_state = random.getstate()
        random.setstate(self._random_state)
        yield
        self._random_state = random.getstate()
        random.setstate(old_state)

    @property
    def random_state(self):
        return deepcopy(self._random_state)

    @random_state.setter
    def random_state(self, s):
        self._random_state = s

    def __call__(self, data):
        """Shuffle and return a new list."""
        with self.use_internal_state():
            return random.sample(data, len(data))


#THIS IS FROM SNEHA  https://github.com/sneha-desai
def stitch_datasets(df_list, on, index_list=None):
    print(index_list)
    # change column names to all caps
    for i in range(len(df_list)):
        df_list[i].columns = map(str.lower, df_list[i].columns)

    # create an empty Dataframe and set first column to on
    merged_df = pd.DataFrame(columns=[on])

    # if indexes are not specified, create an added column for each feature
    # otherwise, only create extra column for features in list
    if index_list is None:
        for i in range(len(df_list)):
            col_list_1 = list(df_list[i].columns)
            df = pd.DataFrame(1, index=df_list[i].index,
                              columns=np.arange(len(df_list[i].columns) - 1))
            col_list_2 = list(df.columns)
            df_list[i] = pd.concat([df_list[i], df], axis=1)
            concat_list = [None] * (len(col_list_1) + len(col_list_2))
            concat_list[0] = col_list_1[0]
            col_list_1 = col_list_1[1:(len(col_list_1))]
            concat_list[1::2] = col_list_1
            concat_list[2::2] = col_list_2
            df_list[i] = df_list[i][concat_list]
    else:
        print(df_list[0])
        print(df_list[1])
        frequency = [0] * len(df_list)
        for j in range(len(df_list)):
            for k in range(len(index_list)):
                for l in range(len(df_list[j].columns)):
                    if (list(df_list[j].columns))[l] == index_list[k]:
                        frequency[j] += 1
        for i in range(len(df_list)):
            if frequency[i] == 0:
                df_list[i] = df_list[i]
            else:
                col_list_1 = list(df_list[i].columns)
                df = pd.DataFrame(1, index=df_list[i].index, columns=np.arange(frequency[i]))
                col_list_2 = list(df.columns)
                df_list[i] = pd.concat([df_list[i], df], axis=1)
                concat_list = [None] * (len(col_list_1) + len(col_list_2))
                concat_list[0] = col_list_1[0]
                col_list_1 = col_list_1[1:(len(col_list_1))]
                concat_list[1::2] = col_list_1
                concat_list[2::2] = col_list_2
                df_list[i] = df_list[i][concat_list]

    for j in range(len(df_list)):
        merged_df = pd.merge(merged_df, df_list[j], how='outer', on=on)

    merged_df.fillna(0, inplace=True)

    return merged_df