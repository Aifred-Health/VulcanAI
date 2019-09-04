# coding=utf-8
""" Defines test cases for tabular dataset """
import pytest
from vulcanai.datasets import tabular_data_utils
import os
import pandas as pd
import torch
import numpy as np

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
        # Nan just an artifact of this dataset.
        return pd.read_csv(fpath, na_values='Nan')

    def test_convert_to_tensor_dataset(self, my_test_dataset):

        only_numeric = my_test_dataset.drop("LowBirthWeight", axis=1)
        res1 = tabular_data_utils.convert_to_tensor_datasets(only_numeric,
                                                             target_vars="mnocig"
                                                             )
        res2 = tabular_data_utils.convert_to_tensor_datasets(only_numeric,
                                                             target_vars="mnocig",
                                                             continuous_target=True)

        assert isinstance(res1, torch.utils.data.TensorDataset)
        assert isinstance(res2, torch.utils.data.TensorDataset)

    def test_create_label_encoding(self, my_test_dataset):
        tabular_data_utils.create_label_encoding(my_test_dataset,
                                                 "LowBirthWeight",
                                              {"Low": 0, "Normal": 1})
        assert set(list(my_test_dataset["LowBirthWeight"].unique())) \
            == {0, 1}

    def test_create_one_hot_encoding(self, my_test_dataset):
        res = tabular_data_utils.create_one_hot_encoding(my_test_dataset,
                                                   "LowBirthWeight")
        assert "LowBirthWeight@Low" in set(list(res.columns))

    def test_reverse_create_all_one_hot_encodings(self, my_test_dataset):
        tabular_data_utils.create_one_hot_encoding(my_test_dataset,
                                                   "LowBirthWeight")
        res = tabular_data_utils.reverse_create_one_hot_encoding(my_test_dataset,
                                                           prefix_sep="@")
        assert "LowBirthWeight@Low" not in set(list(res.columns))

    def test_identify_null(self, my_test_dataset):
        num_threshold = 0.2
        res = tabular_data_utils.identify_null(my_test_dataset, num_threshold)
        assert {'fedyrs'} == set(res)

    def test_identify_unique(self, my_test_dataset):
        res = tabular_data_utils.identify_unique(my_test_dataset, 5)
        assert set(res) == {'headcirumference', 'smoker', 'fedyrs',
                            'LowBirthWeight'}

    def test_identify_unbalanced_columns(self, my_test_dataset):
        res = tabular_data_utils.identify_unbalanced_columns(my_test_dataset,
                                                             0.5)
        assert set(res) == {'headcirumference', 'smoker', 'mnocig',
                            'LowBirthWeight'}

    def test_identify_highly_correlated(self, my_test_dataset):
        res = tabular_data_utils.identify_highly_correlated(my_test_dataset,
                                                            0.2)
        assert (('fage', 'motherage'), 0.8065844173531495) in res

    def test_identify_low_variance(self, my_test_dataset):
        res = tabular_data_utils.identify_low_variance(my_test_dataset, 0.05
                                                       )
        assert 'Gestation' in res


# This is breaking lint for now because it aids in the clarity of the data
# TODO: refactor
class TestStitchDataset:
    @pytest.fixture
    def my_test_dataset_one(self):
        dct_dfs = {'df_test_one': pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'], 'B': ['B0', 'B1', 'B2', 'B3'],
                                                'C': ['C0', 'C1', 'C2', 'C3'], 'D': ['D0', 'D1', 'D2', 'D3']},
                                               index=[0, 1, 2, 3]),
                   'df_test_two': pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'], 'B': ['B4', 'B5', 'B6', 'B7'],
                                                'C': ['C4', 'C5', 'C6', 'C7'], 'D': ['D4', 'D5', 'D6', 'D7']},
                                               index=[4, 5, 6, 7])}
        return dct_dfs

    @pytest.fixture
    def my_test_dataset_two(self):
        dct_dfs = {'df_test_one': pd.DataFrame({'name': ['Jane', 'John', 'Jesse', 'Jane', 'John', 'Jesse'],
                                                'age': [23, 25, 26, np.nan, np.nan, np.nan]},
                                               index=[0, 1, 2, 3, 4, 5]),
                   'df_test_two': pd.DataFrame({'name': ['Jane', 'John', 'Jesse'],
                                                'state': ['CA', 'WA', 'OR']},
                                               index=[0, 1, 2])}
        return dct_dfs

    @pytest.fixture
    def my_test_dataset_three(self):
        dct_dfs = {'df_test_one': pd.DataFrame({'name': ['Jane', 'John', 'Jesse', 'Jane', 'John', 'Jesse', 'Jane'],
                                                'age': [23, 25, 26, np.nan, np.nan, np.nan, 23],
                                                'dob': ['09-18-1995', '10-18-1993', '06-18-1992', np.nan, np.nan,
                                                        np.nan,
                                                        '05-23-1995']},
                                               index=[0, 1, 2, 3, 4, 5, 6]),
                   'df_test_two': pd.DataFrame({'name': ['Jane', 'John', 'Jesse', 'Jane'],
                                                'dob': ['09-18-1995', '10-18-1993', '06-18-1992', '05-23-1995'],
                                                'state': ['CA', 'WA', 'OR', 'AZ']},
                                               index=[0, 1, 2, 3])}
        return dct_dfs

    @pytest.fixture
    def my_test_dataset_four(self):
        dct_dfs = {'df_test_one': pd.DataFrame({'name': ['Jane', 'John', 'Jesse', 'Jane'],
                                                'age': [23, 25, 26, 23],
                                                'dob': ['09-18-1995', '10-18-1993', '06-18-1992', '05-23-1995'],
                                                'visit_date': ['11/29/2018', '11/29/2018', '11/29/2018', '11/29/2018'],
                                                'visit_location': ['HI', 'HI', 'HI', 'HI']},
                                               index=[0, 1, 2, 3]),
                   'df_test_two': pd.DataFrame({'name': ['John', 'Jesse'],
                                                'age': [25, 26],
                                                'dob': ['10-18-1993', '06-18-1992'],
                                                'visit_date': ['09/12/2018', '12/20/2017'],
                                                'visit_location': ['CA', 'AZ']},
                                               index=[0, 1])}
        return dct_dfs

    def test_no_merge_on_columns(self, my_test_dataset_one):
        # MOC (merge on columns)
        df_no_moc_results = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7'],
                                          'B': ['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
                                          'C': ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'],
                                          'D': ['D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']},
                                         index=[0, 1, 2, 3, 4, 5, 6, 7])

        stitch_dataset_results = tabular_data_utils.stitch_datasets(merge_on_columns=None, **my_test_dataset_one)
        pd.testing.assert_frame_equal(stitch_dataset_results, df_no_moc_results)

    def test_single_merge_on_columns(self, my_test_dataset_two):
        df_single_moc_results = pd.DataFrame({'name': ['Jane', 'John', 'Jesse'],
                                              'age': [23, 25, 26],
                                              'state': ['CA', 'WA', 'OR']},
                                             index=[0, 1, 2])
        stitch_dataset_results = tabular_data_utils.stitch_datasets(merge_on_columns=['name'], **my_test_dataset_two)

        # Assert_frame_equal checks order of columns; therefore, sort_index by columns when checking. If dataframes are
        # same, they should sort the same way.
        pd.testing.assert_frame_equal(stitch_dataset_results.sort_index(axis=1), df_single_moc_results.sort_index(axis=1),
                                      check_dtype=False)

    def test_two_merge_on_columns(self, my_test_dataset_three):
        df_two_moc_results = pd.DataFrame({'name': ['Jane', 'John', 'Jesse', 'Jane'],
                                              'age': [23, 25, 26, 23],
                                              'dob': ['09-18-1995', '10-18-1993', '06-18-1992', '05-23-1995'],
                                              'state': ['CA', 'WA', 'OR', 'AZ']},
                                             index=[0, 1, 2, 6])
        stitch_dataset_results = tabular_data_utils.stitch_datasets(merge_on_columns=['name', 'dob'], **my_test_dataset_three)

        # Assert_frame_equal checks order of columns; therefore, sort_index by columns when checking. If dataframes are
        # same, they should sort the same way.
        pd.testing.assert_frame_equal(stitch_dataset_results.sort_index(axis=1), df_two_moc_results.sort_index(axis=1),
                                      check_dtype=False)

    def test_three_merge_on_columns(self, my_test_dataset_four):
        df_three_moc_results = pd.DataFrame({'name': ['Jane', 'John', 'Jesse', 'Jane', 'John', 'Jesse'],
                                             'age': [23, 25, 26, 23, 25, 26],
                                             'dob': ['09-18-1995', '10-18-1993', '06-18-1992', '05-23-1995',
                                                     '10-18-1993', '06-18-1992'],
                                             'visit_date': ['11/29/2018', '11/29/2018', '11/29/2018', '11/29/2018',
                                                            '09/12/2018', '12/20/2017'],
                                             'visit_location': ['HI', 'HI', 'HI', 'HI', 'CA', 'AZ']},
                                            index=[0, 1, 2, 3, 4, 5])

        stitch_dataset_results = tabular_data_utils.stitch_datasets(merge_on_columns=['name', 'dob', 'visit_date'], **my_test_dataset_four)

        # Assert_frame_equal checks order of columns; therefore, sort_index by columns when checking. If dataframes are
        # same, they should sort the same way.
        pd.testing.assert_frame_equal(stitch_dataset_results.sort_index(axis=1), df_three_moc_results.sort_index(axis=1), check_dtype=False)
