# coding=utf-8
""" Defines test cases for tabular dataset """
import pytest
from vulcanai2.datasets import TabularDataset
import os

class TestTabularDataset:

    @pytest.fixture
    def split_data(self):
        pass

    @pytest.fixture
    def my_test_dataset(self):
        """Create a dataset by importing from the test csv"""
        return TabularDataset(
            data="datasets/test_data/birthweight_reduced.csv", #TODO: not sure how to define file path here..
            label_column="id",
            na_values=["Nan"]
        )

    def test_single_dataset_length(self, my_test_dataset):
        assert len(my_test_dataset) == 42
        assert "id" in my_test_dataset.label_column

    def test_save_dataframe(self, my_test_dataset):
        fname = "datasets/test_data/test_save.csv"
        my_test_dataset.save_dataframe(fname)
        assert os.path.isfile(fname)
        os.remove(fname)

    def test_list_columns(self, my_test_dataset):
        column_list = ["id", "headcirumference", "length", "Birthweight", "LowBirthWeight", "Gestation", "smoker", "motherage",
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
        my_test_dataset.create_label_encoding("LowBirthWeight", {"Low": 0, "Normal": 1})
        assert set(my_test_dataset.list_all_column_values("LowBirthWeight")) == {0,1}

    def test_create_one_hot_encoding(self, my_test_dataset):
        my_test_dataset.create_one_hot_encoding("LowBirthWeight")
        assert "LowBirthWeight@Low" in my_test_dataset.list_all_features()

    def test_reverse_create_all_one_hot_encodings(self, my_test_dataset):
        my_test_dataset.create_one_hot_encoding("LowBirthWeight")
        my_test_dataset.reverse_create_all_one_hot_encodings()
        assert "LowBirthWeight@Low" not in my_test_dataset.list_all_features()

    def test_identify_sufficient_non_null(self, my_test_dataset):
        num_threshold = 0.95
        res = my_test_dataset.identify_sufficient_non_null(num_threshold)
        print(res, 'NON NULL')
        assert {'headcirumference', 'fedyrs', 'mnocig'} == set(res)

    def test_identify_unique(self, my_test_dataset):
        res = my_test_dataset.identify_unique(5)
        assert set(res) == {'id','length','Birthweight','Gestation','motherage','mnocig','mheight','mppwt','fage'}

    def test_identify_unbalanced_columns(self, my_test_dataset):
        res = my_test_dataset.identify_unbalanced_columns(0.5)
        assert set(res) == {'headcirumference', 'smoker','mnocig', 'LowBirthWeight'}

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