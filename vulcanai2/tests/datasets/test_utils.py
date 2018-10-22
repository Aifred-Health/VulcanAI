from vulcanai2.datasets.utils import stitch_datasets
import unittest
import pandas as pd


class TestStitchDataset(unittest.TestCase):
    def test_no_merge_on_columns(self):
        dct_dfs = {'df_test_one': pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'], 'B': ['B0', 'B1', 'B2', 'B3'],
                                                'C': ['C0', 'C1', 'C2', 'C3'], 'D': ['D0', 'D1', 'D2', 'D3']},
                                               index=[0, 1, 2, 3]),
                   'df_test_two': pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'], 'B': ['B4', 'B5', 'B6', 'B7'],
                                                'C': ['C4', 'C5', 'C6', 'C7'], 'D': ['D4', 'D5', 'D6', 'D7']},
                                               index=[4, 5, 6, 7])}

        # MOF (merge on columns)
        df_no_moc_results = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7'],
                                          'B': ['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
                                          'C': ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'],
                                          'D': ['D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']},
                                         index=[0, 1, 2, 3, 4, 5, 6, 7])

        stitch_dataset_results = stitch_datasets(dct_dfs, merge_on_columns=None)
        pd.testing.assert_frame_equal(stitch_dataset_results, df_no_moc_results)

if __name__ == '__main__':
    unittest.main()
