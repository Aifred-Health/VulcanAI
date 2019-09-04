# coding=utf-8
"""Includes TestMetrics which tests the metrics class of vulcanai."""
import pytest
import numpy as np
import torch
from vulcanai.models.metrics import Metrics
from vulcanai.models.cnn import ConvNet
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import os


# noinspection PyProtectedMember
class TestMetrics:
    """
    Test metric functionality.

    Note: The number of classes NEEDS to stay consistent
    since the metric values were calculated externally.

    """

    @pytest.fixture
    def metrics(self):
        """Create a metric object fixture to test."""
        # TODO: check that this won't mess with replicability
        np.random.seed(1234)
        return Metrics()

    @pytest.fixture
    def cnn_class(self):
        """Create ConvNet with prediction layer."""
        return ConvNet(
            name='Test_ConvNet_class',
            in_dim=(1, 28, 28),
            config={
                'conv_units': [
                    {
                        "in_channels": 10,
                        "out_channels": 16,
                        "kernel_size": (5, 5),
                        "stride": 2
                    },
                    {
                        "in_channels": 16,
                        "out_channels": 1,
                        "kernel_size": (5, 5),
                        "stride": 1,
                        "padding": 2
                    }]
            },
            num_classes=10
        )

    @pytest.fixture
    def cnn_class_binary(self):
        """Create ConvNet with prediction layer."""
        return ConvNet(
            name='Test_ConvNet_class',
            in_dim=(1, 28, 28),
            config={
                'conv_units': [
                    {
                        "in_channels": 10,
                        "out_channels": 16,
                        "kernel_size": (5, 5),
                        "stride": 2
                    },
                    {
                        "in_channels": 16,
                        "out_channels": 1,
                        "kernel_size": (5, 5),
                        "stride": 1,
                        "padding": 2
                    }]
            },
            num_classes=2
        )

    def create_target_predictions(self):
        """Create target and predictions, used in most metric tests."""
        num_items = 300
        test_target = np.random.randint(0, 10, size=num_items)
        test_predictions = np.random.randint(0, 10, size=num_items)

        return test_target, test_predictions

    def test_extract_class_labels(self, metrics):
        """Correctly represents max likelihood class."""
        test_input = np.array([
                [0.2, 0.8],
                [0.7, 0.3],
                [0.25, 0.75]
            ])
        output = metrics.transform_outputs(test_input)
        assert np.all(output == np.array([1, 0, 1]))

    def test_cross_validate_outputs(self, metrics, cnn_class):
        """Tests that the cross-validate outputs are in the correct form."""
        num_items = 300
        test_input = torch.Tensor(np.random.randint(0, 10,
                                                    size=(num_items,
                                                          *cnn_class.in_dim)))
        test_target = torch.LongTensor(np.random.randint(0, 10,
                                                         size=num_items))
        test_dataloader = DataLoader(TensorDataset(test_input, test_target))

        k = 2
        epochs = 2

        averaged_results = metrics.cross_validate(
            cnn_class, test_dataloader, k, epochs,
            average_results=True)
        all_results = metrics.cross_validate(
            cnn_class, test_dataloader, k, epochs,
            average_results=False)

        for k in averaged_results:
            assert isinstance(averaged_results[k], float)

        for k in all_results:
            assert isinstance(all_results[k], list)

    def test_get_score(self, metrics):
        """Test that get score returns correct values, with complex params."""
        test_target, test_predictions = self.create_target_predictions()

        metrics_to_test = ["sensitivity", "specificity", "accuracy"]

        res_dict = metrics.get_score(test_target, test_predictions,
                                     metrics=metrics_to_test, average="macro",
                                     class_converted=True)

        target_sensitivity = 0.08617573063137578
        np.testing.assert_almost_equal(res_dict["sensitivity"],
                                       target_sensitivity)

        target_specificity = 0.89853925
        np.testing.assert_almost_equal(res_dict["specificity"],
                                       target_specificity)

        target_res = 0.08666666666666667
        np.testing.assert_almost_equal(res_dict["accuracy"],
                                       target_res)

    def test_get_confusion_matrix_values(self, metrics):
        """
        Test get_confusion_matrix_values if it returns the correct values.

        The values being true positives, true negatives, false positives,
        and false negatives (tp, tn, fp, fn, respectively). It will return
        class-specific values.

        """
        test_target, test_predictions = self.create_target_predictions()

        tp, tn, fp, fn = metrics.get_confusion_matrix_values(test_target,
                                                             test_predictions)

        target_tp = [2., 4., 3., 0., 2., 3., 1., 5., 3., 3.]
        target_tn = [249., 231., 247., 238., 247., 244., 248., 246., 248.,
                     228.]
        target_fp = [20., 33., 21., 32., 21., 32., 34., 23., 25., 33.]
        target_fn = [29., 32., 29., 30., 30., 21., 17., 26., 24., 36.]

        np.testing.assert_almost_equal(tp, target_tp)
        np.testing.assert_almost_equal(tn, target_tn)
        np.testing.assert_almost_equal(fp, target_fp)
        np.testing.assert_almost_equal(fn, target_fn)

    def test_check_average_parameter(self, metrics):
        """Test the test average-ing parameter returns error as expected."""
        test_target, test_predictions = self.create_target_predictions()

        metrics._check_average_parameter(test_target, test_predictions,
                                         average="macro")
        # try again with binary data
        num_items = 300
        test_target = np.random.randint(0, 2, size=num_items)
        test_predictions = np.random.randint(0, 2, size=num_items)
        with pytest.raises(ValueError):
            metrics._check_average_parameter(test_target, test_predictions,
                                             average="macro")

    def test_get_sensitivity(self, metrics):
        """Test sensitivity metric calculation."""
        test_target, test_predictions = self.create_target_predictions()

        res = metrics.get_sensitivity(test_target, test_predictions,
                                      average="macro")
        target_res = 0.08617573063137578
        np.testing.assert_almost_equal(res, target_res)

        res = metrics.get_sensitivity(test_target, test_predictions)
        target_res = [0.06451613, 0.11111111, 0.09375, 0., 0.0625,
                      0.125, 0.05555556, 0.16129032, 0.11111111, 0.07692308]
        np.testing.assert_almost_equal(res, target_res)

    def test_get_specificity(self, metrics):
        """Test that specificity returns values as expected."""
        test_target, test_predictions = self.create_target_predictions()

        res = metrics.get_specificity(test_target, test_predictions,
                                      average="macro")
        target_res = 0.89853925
        np.testing.assert_almost_equal(res, target_res)

        res = metrics.get_specificity(test_target, test_predictions)
        target_res = [0.92565054, 0.875, 0.92164177, 0.88148147, 0.92164177,
                      0.884058, 0.8794326, 0.91449815, 0.9084249, 0.87356323]
        np.testing.assert_almost_equal(res, target_res)

    def test_get_dice(self, metrics):
        """Test that dic returns values as expected."""
        test_target, test_predictions = self.create_target_predictions()

        res = metrics.get_dice(test_target, test_predictions,
                               average="macro")
        target_res = 0.086294405
        np.testing.assert_almost_equal(res, target_res)

        res = metrics.get_dice(test_target, test_predictions)
        target_res = [0.0754717, 0.10958904, 0.10714286, 0., 0.07272727,
                      0.10169491, 0.03773585, 0.16949153, 0.10909091, 0.08]
        np.testing.assert_almost_equal(res, target_res)

    def test_get_ppv(self, metrics):
        """Test that ppv returns values as expected."""
        test_target, test_predictions = self.create_target_predictions()

        res = metrics.get_ppv(test_target, test_predictions,
                              average="macro")
        target_res = 0.08943070540896628
        np.testing.assert_almost_equal(res, target_res)

        res = metrics.get_ppv(test_target, test_predictions)
        target_res = [0.09090909, 0.10810811, 0.125, 0., 0.08695652,
                      0.08571429, 0.02857143, 0.17857143, 0.10714286,
                      0.08333333]
        np.testing.assert_almost_equal(res, target_res)

    def test_get_npv(self, metrics):
        """Test that npv returns values as expected."""
        test_target, test_predictions = self.create_target_predictions()

        res = metrics.get_npv(test_target, test_predictions,
                              average="macro")
        target_res = 0.8985111
        np.testing.assert_almost_equal(res, target_res)

        res = metrics.get_npv(test_target, test_predictions)
        target_res = [0.89568347, 0.878327, 0.89492756, 0.8880597, 0.89169675,
                      0.92075473, 0.9358491, 0.9044118, 0.9117647, 0.8636364]
        np.testing.assert_almost_equal(res, target_res)

    def test_get_accuracy(self, metrics):
        """Test that accuracy returns values as expected."""
        test_target, test_predictions = self.create_target_predictions()

        res = metrics.get_accuracy(test_target, test_predictions)
        target_res = 0.08666666666666667
        np.testing.assert_almost_equal(res, target_res)

    def test_get_f1(self, metrics):
        """Test that f1 returns values as expected."""
        test_target, test_predictions = self.create_target_predictions()

        res = metrics.get_f1(test_target, test_predictions,
                             average="macro")
        target_res = 0.08629440679047068
        np.testing.assert_almost_equal(res, target_res)

        res = metrics.get_f1(test_target, test_predictions)
        target_res = [0.0754717, 0.10958904, 0.10714286, 0., 0.07272727,
                      0.10169492, 0.03773585, 0.16949153, 0.10909091, 0.08]

        np.testing.assert_almost_equal(res, target_res)

    def test_get_auc(self, metrics):
        """Test that auc returns values as expected."""
        num_items = 300
        test_target = torch.LongTensor(np.random.randint(0, 10,
                                                         size=num_items))

        raw_predictions = np.random.randint(0, 10, size=(300, 10))

        res = metrics.get_auc(test_target, raw_predictions, num_classes=10)

        target_res = [0.40700323779829717, 0.3952020202020202,
                      0.6441231343283582, 0.5640740740740741,
                      0.4857159514925373, 0.5764643719806763,
                      0.4055358550039401, 0.5535435903585562,
                      0.41982091982091985, 0.48482169171824346]

        np.testing.assert_almost_equal(res, target_res)

    def test_run_test(self, metrics, cnn_class):
        """Test that run_test returns values as expected."""
        num_items = 300

        test_input = torch.Tensor(np.random.randint(0, 10,
                                                    size=(num_items,
                                                          *cnn_class.in_dim)))
        test_target = torch.LongTensor(np.random.randint(0, 10,
                                                         size=num_items))
        test_dataloader = DataLoader(TensorDataset(test_input, test_target))

        res_dict = metrics.run_test(cnn_class, test_dataloader)

        required_metrics = ['accuracy', 'macro_sensitivity',
                            'macro_specificity', 'avg_dice', 'macro_ppv',
                            'macro_npv', 'macro_f1', 'macro_auc']

        assert all(k in res_dict for k in required_metrics)

    def test_run_test_binary(self, metrics, cnn_class_binary):
        """Test that run_test returns values as expected."""
        num_items = 300

        test_input = torch.Tensor(np.random.randint(0, 2,
                                                    size=(num_items,
                                                          *cnn_class_binary.in_dim)))
        test_target = torch.LongTensor(np.random.randint(0, 2,
                                                         size=num_items))

        test_dataloader = DataLoader(TensorDataset(test_input, test_target))

        res_dict = metrics.run_test(cnn_class_binary, test_dataloader)

        required_metrics = ['accuracy', 'macro_sensitivity',
                            'macro_specificity', 'avg_dice', 'macro_ppv',
                            'macro_npv', 'macro_f1', 'macro_auc']

        assert all(k in res_dict for k in required_metrics)

    def test_cross_validate_outputs_stratified(self, metrics, cnn_class):
        """Tests that the cross-validate outputs are in the correct form."""
        num_items = 300
        test_input = torch.Tensor(np.random.randint(0, 10,
                                                    size=(num_items,
                                                          *cnn_class.in_dim)))
        test_target = torch.LongTensor(np.random.randint(0, 10,
                                                         size=num_items))
        test_dataloader = DataLoader(TensorDataset(test_input, test_target))

        k = 2
        epochs = 2

        averaged_results = metrics.cross_validate(
            cnn_class, test_dataloader, k, epochs,
            average_results=True, stratified=True)
        all_results = metrics.cross_validate(
            cnn_class, test_dataloader, k, epochs,
            average_results=False)

        for k in averaged_results:
            assert isinstance(averaged_results[k], float)

        for k in all_results:
            assert isinstance(all_results[k], list)

    def test_stratified_split(self, metrics, cnn_class):
        """Tests that the stratified split function returns a reasonable number
        of datasets"""
        num_items = 300
        test_input = torch.Tensor(np.random.randint(0, 10,
                                                    size=(num_items,
                                                          10)))

        # stratified does not work for complex multi-dimensional data
        test_target = torch.LongTensor(np.random.randint(0, 10,
                                                         size=num_items))
        test_dataset = TensorDataset(test_input, test_target)

        k = 3

        res_target = metrics.stratified_split(test_dataset, k)

        rest_index = metrics.stratified_split(test_dataset, k, 2)

        assert len(res_target) == k
        assert len(rest_index) == k
