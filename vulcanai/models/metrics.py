# coding=utf-8
"""Defines the network test suite."""
import torch
from torch.utils import data

import math
import numpy as np
from sklearn import metrics as skl_metrics

from .utils import round_list
from ..plotters.visualization import display_confusion_matrix
from collections import defaultdict

import copy

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


# noinspection PyProtectedMember
class Metrics(object):
    """
    A class to calculate all metrics for a BaseNetwork.

    Responsible for the test suite.
    """

    def __init__(self):
        """Initialize the metrics class for a BaseNetwork."""

    # TODO: consider making class_converted default True
    @staticmethod
    def get_score(targets, predictions, metrics='accuracy', average=None,
                  class_converted=False):
        """
        Calculate the provided metrics given some targets and predictions.

        Parameters:
            targets: numpy.ndarray of integers
                The target values
            predictions: numpy.ndarray of integers or numpy.ndarray of floats
                The predicted values
            metrics: list of strings
                The strings definition the "get_"... functions to call
            average: string,
                [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
                This parameter is required for multiclass/multilabel targets.
                If None, the scores for each class are returned.
                Otherwise, this determines the type of averaging performed on the
                data.See scikit learn
            class_converted: binary. default False
                True: If raw_predictions have already been converted using
                extract_class_labels
                False: If raw_predictions are used

        Returns:
            metrics: dict
                Values returned by the metric functions.

        """
        double_parametered_functions = ["get_accuracy"]

        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().detach().numpy()

        if not class_converted:
            predictions = Metrics.extract_class_labels(predictions)

        if isinstance(metrics, str):
            metrics = [metrics]

        results_dict = {}

        for metric in metrics:
            method_name = "get_" + metric
            method = getattr(Metrics, method_name)
            try:
                if method_name in set(double_parametered_functions):
                    metric_results = method(targets, predictions)
                else:
                    metric_results = method(targets, predictions, average)

            except (AttributeError, TypeError):
                logger.warning(
                    "Metric {} does not exist via function {}().".format(
                        metric, method_name))

            results_dict[metric] = metric_results

        return results_dict

    @staticmethod
    def extract_class_labels(in_matrix):
        """
        Reformat truth matrix to be the classes in a 1D array.

        Parameters:
            in_matrix : numpy.ndarray or torch.Tensor
                One-hot matrix of shape [batch, num_classes].

        Returns:
            class_list : numpy.ndarray
                1D class array.

        """
        if isinstance(in_matrix, torch.Tensor):
            in_matrix = in_matrix.cpu().detach().numpy()
        # For one-hot encoded entries
        if in_matrix.shape[1] > 1:
            return np.argmax(in_matrix, axis=1)
        # For binary entries
        elif in_matrix.shape[1] == 1:
            return np.around(in_matrix)

    @staticmethod
    def get_confusion_matrix_values(targets, predictions):
        """
        Calculate the tp, tn, fp, fn values given targets and predictions.

        Parameters:
            targets: numpy.ndarray of integers
                The target values.
            predictions: numpy.ndarray of integers
                The predicted values.

        Returns:
            tp, tn, fp, fn:  array of np.float32 with classes in sorted order
                The true positive/negative, false positive/negative values.

        """
        # credit: Robert Fratila
        confusion_matrix = skl_metrics.confusion_matrix(targets, predictions)
        tp = np.diagonal(confusion_matrix).astype('float32')
        tn = (np.array(
            [np.sum(confusion_matrix)] *
            confusion_matrix.shape[0]) -
            confusion_matrix.sum(axis=0) -
            confusion_matrix.sum(axis=1) + tp).astype('float32')
        # sum each column and remove diagonal
        fp = (confusion_matrix.sum(axis=0) - tp).astype('float32')
        # sum each row and remove diagonal
        fn = (confusion_matrix.sum(axis=1) - tp).astype('float32')

        return tp, tn, fp, fn

    @staticmethod
    def _check_average_parameter(targets, predictions, average):
        """
        Check to see if average parameter is suitable for the data.

        Parameters:
            targets: numpy.ndarray of integers
                The target values
            predictions: numpy.ndarray of integers
                The predicted values
            average: string
                [None, ‘binary’ (def), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
                This parameter is required for multiclass/multilabel targets.
                If None, the scores for each class are returned.
                Otherwise, this determines the type of averaging performed on
                the data. See scikit learn.

        Raises:
            ValueError if `average` value invalid for type of predictions.

        Returns:
            boolean

        """
        if not average:
            return True

        if np.unique(predictions).size <= 2 and np.unique(targets).size <= 2:
            if "binary" not in average:
                raise ValueError(
                    "You must provide binary as the average \
                    function if binary data"
                )

        if np.unique(predictions).size >= 2 or np.unique(targets).size >= 2:
            if "binary" in average:
                raise ValueError(
                    "You cannot provide binary as the average \
                    function if non binary data"
                )

        return True

    @staticmethod
    def get_sensitivity(targets, predictions, average=None):
        """
        Calculate the sensitivity. 
        
        Also referred to as recall, or the true positive rate.

        Parameters:
            targets: numpy.ndarray of integers
                The target values.
            predictions: numpy.ndarray of integers
                The predicted values.
            average: string
                [None, ‘binary’ (def), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
                This parameter is required for multiclass/multilabel targets.
                If None, the scores for each class are returned.
                Otherwise, this determines the type of averaging performed on
                the data. See Scikit learn.

        Returns:
            sensitivity: np.float32 or array of np.float32
                The sensitivity.

        """
        assert Metrics._check_average_parameter(targets, predictions, average)
        sensitivity = skl_metrics.recall_score(targets, predictions,
                                               average=average)
        return sensitivity

    @staticmethod
    def get_specificity(targets, predictions, average=None):
        """
        Calculate the specificity.

        Parameters:
            targets: numpy.ndarray of integers
                The target values.
            predictions: numpy.ndarray of integers
                The predicted values.
            average: string
                [None, ‘binary’ (def), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
                This parameter is required for multiclass/multilabel targets.
                If None, the scores for each class are returned.
                Otherwise, this determines the type of averaging performed on
                the data. See Scikit learn.

        Returns:
            specificity: np.float32 or array of np.float32
                The specificity.

        """
        _, tn, fp, _ = Metrics.get_confusion_matrix_values(targets,
                                                           predictions)
        specificity = tn / (tn + fp)

        # TODO: implement other options
        if average == "macro":
            specificity = np.average(specificity)
        elif average:
            raise NotImplementedError

        return specificity

    @staticmethod
    def get_dice(targets, predictions, average=None):
        """
        Calculate the dice metric.

        Parameters:
            targets: numpy.ndarray of integers
                The target values.
            predictions: numpy.ndarray of integers
                The predicted values.
            average: string
                [None, ‘binary’ (def), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
                Only None and 'macro' currently implemented.
                This parameter is required for multiclass/multilabel targets.
                If None, the scores for each class are returned.
                Otherwise, this determines the type of averaging performed on
                the data. See Scikit learn.

        Returns:
            dice: np.float32 or array of np.float32
                The dice metric.

        """
        tp, _, fp, fn = Metrics.get_confusion_matrix_values(targets,
                                                            predictions)
        dice = 2 * tp / (2 * tp + fp + fn)

        # TODO: implement other options
        if average == "macro":
            dice = np.average(dice)
        elif average:
            raise NotImplementedError

        return dice

    @staticmethod
    def get_ppv(targets, predictions, average=None):
        """
        Calculate the positive predictive value.

        Parameters:
            targets: numpy.ndarray of integers
                The target values.
            predictions: numpy.ndarray of integers
                The predicted values.
            average: string
                [None, ‘binary’ (def), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
                Only None and 'macro' currently implemented.
                This parameter is required for multiclass/multilabel targets.
                If None, the scores for each class are returned.
                Otherwise, this determines the type of averaging performed on
                the data. See Scikit learn.

        Returns:
            ppv: float32 or array of np.float32
                The positive predictive value.

        """
        assert Metrics._check_average_parameter(targets, predictions,
                                                average=average)
        ppv = skl_metrics.precision_score(targets, predictions,
                                          average=average)

        return ppv

    @staticmethod
    def get_npv(targets, predictions, average=None):
        """
        Calculate the negative predictive value.

        Parameters:
            targets: numpy.ndarray of integers
                The target values.
            predictions: numpy.ndarray of integers
                The predicted values.
            average: string
                [None, ‘binary’ (def), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
                Only None and 'macro' currently implemented.
                This parameter is required for multiclass/multilabel targets.
                If None, the scores for each class are returned.
                Otherwise, this determines the type of averaging performed on
                the data. See Scikit learn.

        Returns:
            npv: np.float32 or array of np.float32
                The negative predictive value

        """
        _, tn, _, fn = Metrics.get_confusion_matrix_values(targets,
                                                           predictions)

        npv = np.nan_to_num(tn / (tn + fn))

        # TODO: implement other options
        if average == "macro":
            npv = np.average(npv)
        elif average:
            raise NotImplementedError

        return npv

    @staticmethod
    def get_accuracy(targets, predictions):
        """
        Calculate the accuracy.

        Parameters:
            targets: numpy.ndarray of integers
                The target values.
            predictions: numpy.ndarray of integers
                The predicted values.

        Returns:
            accuracy: array of np.float32
                The accuracy.

        """
        accuracy = skl_metrics.accuracy_score(targets, predictions)

        return accuracy

    @staticmethod
    def get_f1(targets, predictions, average=None):
        """
        Calculate the f1 score.

        Parameters:
            targets: numpy.ndarray of integers
                The target values
            predictions: numpy.ndarray of integers
                The predicted values
            average: string
                [None, ‘binary’ (def), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
                This parameter is required for multiclass/multilabel targets.
                If None, the scores for each class are returned.
                Otherwise, this determines the type of averaging performed on
                the data. See Scikit learn.

        Returns:
            f1: np.float32 or array of np.float32
                The f1 score.

        """
        assert Metrics._check_average_parameter(targets, predictions, average)
        f1 = skl_metrics.f1_score(targets, predictions, average=average)

        return f1

    # TODO: what type are the raw predicted values that come out??
    @staticmethod
    def get_auc(targets, raw_predictions, num_classes, average=None):
        """
        Calculate the AUC. Note: raw_predictions and num_classes are required.

        Parameters:
            targets: numpy.ndarray of integers
                The target values.
            raw_predictions: numpy.ndarray of floats
                The raw predicted values, not converted to classes.
            average: string
                [None, ‘binary’ (def), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
                This parameter is required for multiclass/multilabel targets.
                If None, the scores for each class are returned.
                Otherwise, this determines the type of averaging performed on
                the data. See Scikit learn.

        Returns:
            f1: np.float32 or array of np.float32
                The AUC.

        """
        if raw_predictions.ndim == 1:
            raise ValueError("You must provide raw predictions not \
                             class_converted predictions")

        all_class_auc = []
        for i in range(num_classes):
            if num_classes == 1:
                fpr, tpr, _ = skl_metrics.roc_curve(targets,
                                                    raw_predictions,
                                                    pos_label=1)
            else:
                fpr, tpr, _ = skl_metrics.roc_curve(targets,
                                                    raw_predictions[:, i],
                                                    pos_label=i)
            auc = skl_metrics.auc(fpr, tpr)
            all_class_auc += [auc]
        # TODO: implement other options
        if average == "macro":
            all_class_auc = np.average(all_class_auc)
        elif average:
            raise NotImplementedError

        return all_class_auc


    @staticmethod
    def run_test(network, data_loader, plot=False, save_path=None):
        """
        Will conduct the test suite to determine network strength.

        Parameters:
            data_loader : DataLoader
                A DataLoader object to run the test with.
            save_path : string
                Folder to place images in.
            plot: bool
                Determine if graphs should be plotted in real time.

        Returns:
            results : dict

        """
        if plot:
            logger.setLevel(logging.INFO)

        if network._num_classes is None or \
           network._num_classes == 0:
            raise ValueError('There\'s no classification layer')

        num_classes = network._num_classes
        # getting just the y values out of the dataset
        # TODO: store in tensor for continuity?
        targets = np.array([v[1] for v in data_loader.dataset])

        raw_predictions = network.forward_pass(
            data_loader=data_loader,
            convert_to_class=False)

        predictions = Metrics.extract_class_labels(raw_predictions)

        cm = skl_metrics.confusion_matrix(targets, predictions)
        if plot:
            display_confusion_matrix(cm, save_path=save_path)

        tp, tn, fp, fn = Metrics.get_confusion_matrix_values(targets,
                                                             predictions)

        sensitivity = Metrics.get_specificity(targets, predictions)
        sensitivity_macro = Metrics.get_specificity(targets, predictions,
                                                    average="macro")

        specificity = Metrics.get_specificity(targets, predictions)
        specificity_macro = Metrics.get_specificity(targets, predictions,
                                                    average="macro")

        dice = Metrics.get_dice(targets, predictions)
        dice_macro = Metrics.get_dice(targets, predictions, average="macro")

        ppv = Metrics.get_ppv(targets, predictions)
        ppv_macro = Metrics.get_ppv(targets, predictions, average="macro")

        npv = Metrics.get_npv(targets, predictions)
        npv_macro = Metrics.get_npv(targets, predictions, average="macro")

        accuracy = Metrics.get_accuracy(targets, predictions)

        f1 = Metrics.get_f1(targets, predictions)
        f1_macro = Metrics.get_f1(targets, predictions, average="macro")

        auc = Metrics.get_auc(targets, raw_predictions, num_classes)
        auc_macro = Metrics.get_auc(targets, raw_predictions, num_classes,
                                    average="macro")

        logger.info('{} test\'s results'.format(network.name))

        logger.info('TP: {}'.format(tp))
        logger.info('FP: {}'.format(fp))
        logger.info('TN: {}'.format(tn))
        logger.info('FN: {}'.format(fn))

        logger.info('\nAccuracy: {}'.format(accuracy))

        logger.info('Sensitivity: {}'.format(round_list(sensitivity,
                                                        decimals=3)))
        logger.info('\tMacro Sensitivity: {:.4f}'.format(sensitivity_macro))

        logger.info('Specificity: {}'.format(round_list(specificity,
                                                        decimals=3)))
        logger.info('\tMacro Specificity: {:.4f}'.format(specificity_macro))

        logger.info('DICE: {}'.format(round_list(dice, decimals=3)))
        logger.info('\tAvg. DICE: {:.4f}'.format(np.average(dice)))

        logger.info('Positive Predictive Value: {}'.format(
            round_list(ppv, decimals=3)))
        logger.info('\tMacro Positive Predictive Value: {:.4f}'.format(
            ppv_macro))

        logger.info('Negative Predictive Value: {}'.format(
            round_list(npv, decimals=3)))
        logger.info('\tMacro Negative Predictive Value: {:.4f}'.format(
            npv_macro))

        logger.info('F1-score: {}'.format(round_list(f1, decimals=3)))
        logger.info('\tMacro f1-score: {:.4f}'.format(f1_macro))

        logger.info('auc: {}'.format(round_list(auc, decimals=3)))
        logger.info('\tMacro auc: {:.4f}'.format(auc_macro))

        return {
            'accuracy': float(accuracy),
            'macro_sensitivity': float(sensitivity_macro),
            'macro_specificity': float(specificity_macro),
            'avg_dice': float(dice_macro),
            'macro_ppv': float(ppv_macro),
            'macro_npv': float(npv_macro),
            'macro_f1': float(f1_macro),
            'macro_auc': float(auc_macro)
        }

    # TODO: include support
    @staticmethod
    def cross_validate(network, data_loader, k, epochs,
                       average_results=True, retain_graph=None,
                       valid_interv=4, plot=False, save_path=None):
        """
        Perform k-fold cross validation given a Network and DataLoader object.

        Parameters:
            network : BaseNetwork
                Network descendant of BaseNetwork.
            data_loader : torch.utils.data.DataLoader
                The DataLoader object containing the totality of the data to use
                for k-fold cross validation.
            k : int
                The number of folds to split the training into.
            epochs : int
                The number of epochs to train the network per fold.
            average_results : boolean
                Whether or not to return results from all folds or just an average.
            retain_graph : {None, boolean}
                Whether retain_graph will be true when .backwards is called.
            valid_interv : int
                Specifies after how many epochs validation should occur.
            plot : boolean
                Whether or not to plot all results in prompt and charts.
            save_path : str
                Where to save all figures and results.

        Returns:
            results : dict
                If average_results is on, return dict of floats.
                If average_results is off, return dict of float lists.

        """
        all_results = defaultdict(lambda: [])

        # TODO: this whole section is really clunky
        # Getting the fold sequence.
        fold_len = math.floor(len(data_loader.dataset) / k)
        rem = len(data_loader.dataset) % k
        fold_seq = []

        for _ in range(k-1):
            fold_seq.append(fold_len)
        if rem == 0:
            fold_seq.append(fold_len)
        else:
            fold_seq.append(fold_len+rem)  # last one is the longest if unequal

        dataset_splits = data.random_split(data_loader.dataset,
                                           fold_seq)

        batch_size = data_loader.batch_size

        # #TODO: improve the copying of parameters
        # Set to true if RandomSampler exists.
        shuffle = isinstance(data_loader.sampler, data.sampler.RandomSampler)

        try:
            for fold in range(k):

                # TODO: this may break on different devices?? test.
                # TODO: Re-initialize instead of deepcopy?
                cross_val_network = copy.deepcopy(network)

                # TODO: properly pass params

                # Generate fold training set.
                train_dataset = data.ConcatDataset(
                    dataset_splits[:fold] + dataset_splits[fold+1:])
                # Generate fold validation set.
                val_dataset = dataset_splits[fold]
                # Generate fold training data loader object.
                train_loader = data.DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=shuffle)
                # Generate fold validation data loader object.
                val_loader = data.DataLoader(
                    val_dataset, batch_size=batch_size)
                # Train network on fold training data loader.
                cross_val_network.fit(
                    train_loader, val_loader, epochs,
                    retain_graph=retain_graph,
                    valid_interv=valid_interv, plot=plot, save_path=save_path)
                # Validate network performance on validation data loader.
                results = Metrics.run_test(
                    cross_val_network, val_loader,
                    save_path=save_path, plot=plot)

                logger.info(results)
                for m in results:
                    all_results[m].append(results[m])

        # TODO: we could show something better here like calculate
        # all the results so far
        except KeyboardInterrupt:
            logger.info(
                "\n\n***KeyboardInterrupt: Cross validate stopped \
                prematurely.***\n\n")

        if average_results:
            averaged_all_results = {}
            for m in all_results:
                averaged_all_results[m] = np.mean(all_results[m])
            return averaged_all_results
        else:
            return all_results
