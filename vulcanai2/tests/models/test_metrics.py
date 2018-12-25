"""Define the tests for all metrics."""
import pytest
import numpy as np
import torch
from vulcanai2.models.metrics import Metrics
from vulcanai2.models.cnn import ConvNet
from vulcanai2.models.dnn import DenseNet
from vulcanai2.models.ensemble import SnapshotNet
from torch.utils.data import TensorDataset, DataLoader


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

    def test_extract_class_labels(self, metrics):
        """Correctly represents max likelihood class."""
        test_input = np.array([
                [0.2, 0.8],
                [0.7, 0.3],
                [0.25, 0.75]
            ])
        output = metrics.extract_class_labels(test_input)
        assert np.all(output == np.array([1, 0, 1]))

    def test_cross_validate_outputs(self, metrics, cnn_class):
        """Tests that the cross-validate outputs are in the correct form."""
        num_items = 300

        test_input = torch.Tensor(
            np.random.randint(0, 9, size=(num_items, *cnn_class.in_dim)))
        test_target = torch.LongTensor(np.random.randint(0, 9, size=num_items))
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
        """Test the get_score function for some arbitrary set of metrics."""
        num_items = 300
        test_target = np.random.randint(0, 9, size=num_items)
        test_predictions = np.random.randint(0, 9, size=num_items)

        metrics_to_test = ["sensitivity", "specificity", "accuracy"]

        res_dict = metrics.get_score(test_target, test_predictions,
                                     metrics=metrics_to_test, average="macro",
                                     class_converted=True)

        target_sensitivity = 0.08951250093419325
        np.testing.assert_almost_equal(res_dict["sensitivity"],
                                       target_sensitivity)

        target_specificity = 0.88581157
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
        num_items = 300
        test_target = np.random.randint(0, 9, size=num_items)
        test_predictions = np.random.randint(0, 9, size=num_items)

        tp, tn, fp, fn = metrics.get_confusion_matrix_values(test_target,
                                                             test_predictions)

        target_tp = [2., 4., 2., 2., 3., 4., 4., 3., 2.]
        target_tn = [241., 226., 236., 226., 241., 233., 246., 237., 240.]
        target_fp = [25., 33., 28., 37., 24., 39., 28., 29., 31.]
        target_fn = [32., 37., 34., 35., 32., 24., 22., 31., 27.]

        np.testing.assert_almost_equal(tp, target_tp)
        np.testing.assert_almost_equal(tn, target_tn)
        np.testing.assert_almost_equal(fp, target_fp)
        np.testing.assert_almost_equal(fn, target_fn)

    def test_check_average_parameter(self, metrics):
        """Test average parameter get used properly for binary/multi-class."""
        num_items = 10
        test_target = np.random.randint(0, 9, size=num_items)
        test_predictions = np.random.randint(0, 9, size=num_items)

        metrics._check_average_parameter(test_target, test_predictions,
                                         average="macro")

        num_items = 10
        test_target = np.random.randint(0, 2, size=num_items)
        test_predictions = np.random.randint(0, 2, size=num_items)
        with pytest.raises(ValueError) as e_info:
            metrics._check_average_parameter(test_target, test_predictions,
                                             average="macro")

    def test_get_sensitivity(self, metrics):
        """Test sensitivity metric calculation."""
        num_items = 300
        test_target = np.random.randint(0, 9, size=num_items)
        test_predictions = np.random.randint(0, 9, size=num_items)

        res = metrics.get_sensitivity(test_target, test_predictions,
                                      average="macro")
        target_res = 0.08951250093419325
        np.testing.assert_almost_equal(res, target_res)

        res = metrics.get_sensitivity(test_target, test_predictions)
        target_res = [0.05882353, 0.09756098, 0.05555556, 0.05405405,
                      0.08571429, 0.14285714, 0.15384615, 0.08823529,
                      0.06896552]
        np.testing.assert_almost_equal(res, target_res)

    def test_get_specificity(self, metrics):
        """Test specificity metric calculation."""
        num_items = 300
        test_target = np.random.randint(0, 9, size=num_items)
        test_predictions = np.random.randint(0, 9, size=num_items)

        res = metrics.get_specificity(test_target, test_predictions,
                                      average="macro")
        target_res = 0.88581157
        np.testing.assert_almost_equal(res, target_res)

        res = metrics.get_specificity(test_target, test_predictions)
        target_res = [0.90601504, 0.87258685, 0.8939394, 0.8593156,
                      0.90943396, 0.8566176, 0.8978102, 0.89097744,
                      0.88560885]
        np.testing.assert_almost_equal(res, target_res)

    def test_get_dice(self, metrics):
        """Test dice metric calculation."""
        num_items = 300
        test_target = np.random.randint(0, 9, size=num_items)
        test_predictions = np.random.randint(0, 9, size=num_items)

        res = metrics.get_dice(test_target, test_predictions,
                               average="macro")
        target_res = 0.08713134
        np.testing.assert_almost_equal(res, target_res)

        res = metrics.get_dice(test_target, test_predictions)
        target_res = [0.06557377, 0.1025641, 0.06060606, 0.05263158,
                      0.09677419, 0.11267605, 0.13793103, 0.09090909,
                      0.06451613]
        np.testing.assert_almost_equal(res, target_res)

    def test_get_ppv(self, metrics):
        """Test positive predictive value metric calculation."""
        num_items = 300
        test_target = np.random.randint(0, 9, size=num_items)
        test_predictions = np.random.randint(0, 9, size=num_items)

        res = metrics.get_ppv(test_target, test_predictions,
                              average="macro")
        target_res = 0.08706903640689172
        np.testing.assert_almost_equal(res, target_res)

        res = metrics.get_ppv(test_target, test_predictions)
        target_res = [0.07407407, 0.10810811, 0.06666667, 0.05128205,
                      0.11111111, 0.09302326, 0.125, 0.09375, 0.06060606]
        np.testing.assert_almost_equal(res, target_res)

    def test_get_npv(self, metrics):
        """Test negative predictive value metric calculation."""
        num_items = 300
        test_target = np.random.randint(0, 9, size=num_items)
        test_predictions = np.random.randint(0, 9, size=num_items)

        res = metrics.get_npv(test_target, test_predictions,
                              average="macro")
        target_res = 0.88584304
        np.testing.assert_almost_equal(res, target_res)

        res = metrics.get_npv(test_target, test_predictions)
        target_res = [0.8827839, 0.8593156, 0.8740741, 0.8659004, 0.8827839,
                      0.9066148, 0.91791046, 0.88432837, 0.8988764]
        np.testing.assert_almost_equal(res, target_res)

    def test_get_accuracy(self, metrics):
        """Test accuracy metric calculation."""
        num_items = 300
        test_target = np.random.randint(0, 9, size=num_items)
        test_predictions = np.random.randint(0, 9, size=num_items)

        res = metrics.get_accuracy(test_target, test_predictions)
        target_res = 0.08666666666666667
        np.testing.assert_almost_equal(res, target_res)

    def test_get_f1(self, metrics):
        """Test f1 metric calculation."""
        num_items = 300
        test_target = np.random.randint(0, 9, size=num_items)
        test_predictions = np.random.randint(0, 9, size=num_items)

        res = metrics.get_f1(test_target, test_predictions,
                             average="macro")
        target_res = 0.08713133521331753
        np.testing.assert_almost_equal(res, target_res)

        res = metrics.get_f1(test_target, test_predictions)
        target_res = [0.06557377, 0.1025641, 0.06060606, 0.05263158,
                      0.09677419, 0.11267606, 0.13793103, 0.09090909,
                      0.06451613]
        np.testing.assert_almost_equal(res, target_res)

    def test_get_auc(self, metrics):
        """Test AUC metric calculation."""
        num_items = 300
        test_target = torch.LongTensor(np.random.randint(0, 9, size=num_items))

        raw_predictions = np.random.randint(0, 9, size=(num_items, 10))
        res = metrics.get_f1(test_target, raw_predictions,
                             average="macro")

    def test_run_test(self, metrics):
        """Test the vulcan test suite via run_test."""
        pass
