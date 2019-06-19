# coding=utf-8
""" Defines test cases for tabular dataset """
import pytest
from vulcanai.datasets import TabularDataset
import os
import pandas as pd


# noinspection PyMissingOrEmptyDocstring
class TestTabularDataset:

    @pytest.fixture
    def split_data(self):
        pass

    @pytest.fixture
    def my_test_dataset(self):
        """Create a dataset by importing from the test csv"""
        fpath = str(os.path.dirname(__file__)) + \
            "/test_data/birthweight_reduced.csv"
        return TabularDataset(
            data=fpath,
            label_column="id",
            na_values='Nan'
        )

    @pytest.fixture
    def my_test_dataset_two(self):
        """Create a second dataset by importing from the test csv"""
        fpath = str(os.path.dirname(__file__)) + \
            "/test_data/birthweight_reduced2.csv"
        return fpath

    @pytest.fixture
    def my_merged_test_dataset(self):
        fpath = str(os.path.dirname(__file__)) + \
            "/test_data/birthweight_reduced_merged.csv"
        return pd.read_csv(fpath)

    def test_merge_dataframe(self, my_test_dataset, my_test_dataset_two,
                             my_merged_test_dataset):
        dct_df = {'df_test_two': my_test_dataset_two}
        my_test_dataset.merge_dataframe(merge_on_columns=['id'],
                                        na_values='Nan', **dct_df)
        pd.testing.assert_frame_equal(my_test_dataset.df,
                                      my_merged_test_dataset,
                                      check_dtype=False)

    def test_single_dataset_length(self, my_test_dataset):
        assert len(my_test_dataset) == 42
        assert "id" in my_test_dataset.label_column

    def test_save_dataframe(self, my_test_dataset):
        fpath = str(os.path.dirname(__file__)) + "/test_data/test_save.csv"
        my_test_dataset.save_dataframe(fpath)
        assert os.path.isfile(fpath)
        os.remove(fpath)

    def test_list_columns(self, my_test_dataset):
        column_list = ["id", "headcirumference", "length", "Birthweight",
                       "LowBirthWeight", "Gestation", "smoker", "motherage",
                       "mnocig", "mheight", "mppwt", "fage", "fedyrs"]
        assert set(my_test_dataset.list_all_features()) == set(column_list)

    def test_replace_value_in_column(self, my_test_dataset):
        before = my_test_dataset.list_all_column_values("motherage")
        my_test_dataset.replace_value_in_column("motherage", 24, 25)
        after = my_test_dataset.list_all_column_values("motherage")
        assert (set(before) - set(after)) == {24}

    def test_delete_columns(self, my_test_dataset):
        before = my_test_dataset.list_all_features()
        my_test_dataset.delete_column("motherage")
        after = my_test_dataset.list_all_features()
        assert (set(before) - set(after)) == {"motherage"}

    def test_create_label_encoding(self, my_test_dataset):
        my_test_dataset.create_label_encoding("LowBirthWeight",
                                              {"Low": 0, "Normal": 1})
        assert set(my_test_dataset.list_all_column_values("LowBirthWeight")) \
            == {0, 1}

    def test_create_one_hot_encoding(self, my_test_dataset):
        my_test_dataset.create_one_hot_encoding("LowBirthWeight")
        assert "LowBirthWeight@Low" in my_test_dataset.list_all_features()

    def test_reverse_create_all_one_hot_encodings(self, my_test_dataset):
        my_test_dataset.create_one_hot_encoding("LowBirthWeight")
        my_test_dataset.reverse_create_one_hot_encoding(column_name=None)
        assert "LowBirthWeight@Low" not in my_test_dataset.list_all_features()

    def test_identify_null(self, my_test_dataset):
        num_threshold = 0.2
        res = my_test_dataset.identify_null(num_threshold)
        assert {'fedyrs'} == set(res)

    def test_identify_unique(self, my_test_dataset):
        res = my_test_dataset.identify_unique(5)
        assert set(res) == {'headcirumference', 'smoker', 'fedyrs',
                            'LowBirthWeight'}

    def test_identify_unbalanced_columns(self, my_test_dataset):
        res = my_test_dataset.identify_unbalanced_columns(0.5)
        assert set(res) == {'headcirumference', 'smoker', 'mnocig',
                            'LowBirthWeight'}

    def test_identify_highly_correlated(self, my_test_dataset):
        res = my_test_dataset.identify_highly_correlated(0.2)
        assert (('fage', 'motherage'), 0.8065844173531495) in res

    def test_identify_low_variance(self, my_test_dataset):
        res = my_test_dataset.identify_low_variance(0.05)
        assert 'Gestation' in res

    def test_split_length_correct(self, my_test_dataset):
        res = my_test_dataset.split([0.1, 0.2, 0.7])
        assert len(res) == 3
        assert isinstance(res[1], TabularDataset)
        res = my_test_dataset.split([0.1, 0.9])
        assert len(res) == 2
        assert isinstance(res[1], TabularDataset)

    def test_split_length_correct_stratified(self, my_test_dataset):
        res = my_test_dataset.split([0.1, 0.2, 0.7], stratified=True,
                                    stratum_column="motherage")
        assert len(res) == 3
        assert isinstance(res[1], TabularDataset)
        res = my_test_dataset.split([0.1, 0.9], stratified=True,
                                    stratum_column="motherage")
        assert len(res) == 2
        assert isinstance(res[1], TabularDataset)

    def test_stratified_values(self, my_test_dataset):
        res = my_test_dataset.split([0.1, 0.2, 0.7], stratified=True,
                                    stratum_column="motherage")

        assert list(res[0].df["motherage"].values) == [20]
        assert list(res[1].df["motherage"].values) == [19, 20, 21, 24, 27, 31]
        assert list(res[2].df["motherage"].values) == [18, 19, 19, 20, 20, 20,
                                                       20, 20, 21, 21, 22, 22,
                                                       23, 23, 24, 24, 24, 26,
                                                       26, 27, 27, 27, 28, 28,
                                                       29, 29, 30, 30, 31, 31,
                                                       32, 35, 37, 37, 41]
