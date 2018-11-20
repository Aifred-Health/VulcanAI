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
            data="test_data/birthweight_reduced.csv",
            label_column="id",
        )

    def test_single_dataset_length(self, my_test_dataset):
        assert len(my_test_dataset) == 42
        assert "id" in my_test_dataset.label_column

    def test_save_dataframe(self, my_test_dataset):
        fname = "test_data/test_save.csv"
        my_test_dataset.save_dataframe(fname)
        assert os.path.isfile(fname)
        os.remove(fname)

    def test_list_columns(self, my_test_dataset):
        column_list = ["id", "headcirumference", "length", "Birthweight", "LowBirthWeight", "Gestation", "smoker", "motherage",
                       "mnocig", "mheight", "mppwt", "fage", "fedyrs"]
        assert set(my_test_dataset.list_all_features()) == set(column_list)

    # # TODO: test all possible combinations of merging
    # def test_merge_data(self, my_test_dataset):
    #     data = "test_data/birthweight_reduced2.csv"
    #     my_test_dataset.merge_data(data, "id")
    #     #TODO: test what Jospeh updates

    def test_replace_value_in_column(self, my_test_dataset):

        before = my_test_dataset.list_all_column_values("motherage")
        my_test_dataset.replace_value_in_column("motherage", 24, 25)
        after = my_test_dataset.list_all_column_values("motherage")

        assert set(before) - set(after) == set(25)