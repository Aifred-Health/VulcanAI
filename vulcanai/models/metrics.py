# coding=utf-8
"""Defines the network test suite."""
import torch
from torch.utils import data

import math
import numpy as np
from sklearn import metrics as skl_metrics
import pandas as pd

from .utils import round_list, _get_probs, _filter_matched_subj
from ..plotters.visualization import display_confusion_matrix
from collections import defaultdict

import copy
import inspect

import pandas as pd

from torch.utils.data import Dataset, DataLoader

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
                [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’,
                ‘weighted’]
                This parameter is required for multiclass/multilabel targets.
                If None, the scores for each class are returned.
                Otherwise, this determines the type of averaging performed on
                the data. See scikit learn documentation for each metric.
            class_converted: binary. default False
                True: If raw_predictions have already been converted using
                transform_outputs
                False: If raw_predictions are used

        Returns:
            metrics: dict
                Values returned by the metric functions.

        """
        functions_without_average_parameter = ["get_accuracy", "get_mse"]

        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().detach().numpy()

        if not class_converted:
            predictions = Metrics.transform_outputs(predictions)

        if isinstance(metrics, str):
            metrics = [metrics]

        results_dict = {}

        for metric in metrics:
            method_name = "get_" + metric
            method = getattr(Metrics, method_name)
            try:
                if method_name in set(functions_without_average_parameter):
                    metric_results = method(targets, predictions)
                else:
                    metric_results = method(targets, predictions, average)
                results_dict[metric] = metric_results
            except (AttributeError, TypeError):
                logger.warning(
                    "Metric {} does not exist via function {}().".format(
                        metric, method_name))

        return results_dict

    @staticmethod
    def transform_outputs(in_matrix):
        """
        Reformat output matrix.

        If one-hot, truth matrix to be the classes in a 1D array.

        Note: This does not handle multiple class prediction.

        Parameters:
            in_matrix : numpy.ndarray or torch.Tensor
                One-hot matrix of shape [batch, num_classes].

        Returns:
            class_list : numpy.ndarray of floats
                1D class array.

        """
        if isinstance(in_matrix, torch.Tensor):
            in_matrix = in_matrix.cpu().detach().numpy()

        # For one-hot encoded entries
        if in_matrix.shape[1] > 1:
            return np.argmax(in_matrix, axis=1).astype(np.float32)
        # For single value entries
        elif in_matrix.shape[1] == 1:
            return in_matrix
        else:
            return None

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

        if np.unique(predictions).size > 2 or np.unique(targets).size > 2:
            if "binary" in average:
                raise ValueError(
                    "You cannot provide binary as the average \
                    function if non binary data"
                )

        return True

    @staticmethod
    def get_sensitivity(targets, predictions, average=None, pos_label=1):
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
            pos_label: int
                0 or 1. Only used in the binary case to indicate which class
                label will be used to provide the output for average=binary.


        Returns:
            sensitivity: np.float32 or array of np.float32
                The sensitivity.

        """
        assert Metrics._check_average_parameter(targets, predictions, average)
        sensitivity = skl_metrics.recall_score(targets, predictions,
                                               average=average,
                                               pos_label=pos_label)
        return sensitivity

    @staticmethod
    def get_specificity(targets, predictions, average=None, pos_label=1):
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
            pos_label: int
                0 or 1. Only used in the binary case to indicate which class
                label will be used to provide the output for average=binary.

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
        elif average == "binary":
            specificity = specificity[pos_label]
        elif average:
            raise NotImplementedError

        return specificity

    @staticmethod
    def get_dice(targets, predictions, average=None, pos_label=1):
        """
        Calculate the dice metric.

        Parameters:
            targets: numpy.ndarray of integers
                The target values.
            predictions: numpy.ndarray of integers
                The predicted values.
            average: string
                [None, ‘binary’ (def), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
                Only None, 'binary', and 'macro' currently implemented.
                This parameter is required for multiclass/multilabel targets.
                If None, the scores for each class are returned.
                Otherwise, this determines the type of averaging performed on
                the data. See Scikit learn.
            pos_label: int
                0 or 1. Only used in the binary case to indicate which class
                label will be used to provide the output for average=binary.

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
        elif average == "binary":
            dice = dice[pos_label]
        elif average:
            raise NotImplementedError

        return dice

    @staticmethod
    def get_ppv(targets, predictions, average=None, pos_label=1):
        """
        Calculate the positive predictive value.

        Parameters:
            targets: numpy.ndarray of integers
                The target values.
            predictions: numpy.ndarray of integers
                The predicted values.
            average: string
                [None, ‘binary’ (def), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
                Only None, 'binary', and 'macro' currently implemented.
                This parameter is required for multiclass/multilabel targets.
                If None, the scores for each class are returned.
                Otherwise, this determines the type of averaging performed on
                the data. See Scikit learn.
            pos_label: int
                0 or 1. Only used in the binary case to indicate which class
                label will be used to provide the output for average=binary.

        Returns:
            ppv: float32 or array of np.float32
                The positive predictive value.

        """
        assert Metrics._check_average_parameter(targets, predictions,
                                                average=average)
        ppv = skl_metrics.precision_score(targets, predictions,
                                          average=average, pos_label=pos_label)
        return ppv

    @staticmethod
    def get_npv(targets, predictions, average=None, pos_label=1):
        """
        Calculate the negative predictive value.

        Parameters:
            targets: numpy.ndarray of integers
                The target values.
            predictions: numpy.ndarray of integers
                The predicted values.
            average: string
                [None, ‘binary’ (def), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
                Only None, 'binary', and 'macro' currently implemented.
                This parameter is required for multiclass/multilabel targets.
                If None, the scores for each class are returned.
                Otherwise, this determines the type of averaging performed on
                the data. See Scikit learn.
            pos_label: int
                0 or 1. Only used in the binary case to indicate which class
                label will be used to provide the output for average=binary.

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
        elif average == "binary":
            npv = npv[pos_label]
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
    def get_f1(targets, predictions, average=None, pos_label=1):
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
            pos_label: int default 1:
                The index value that is positive when in the binary case

        Returns:
            f1: np.float32 or array of np.float32
                The f1 score.

        """
        assert Metrics._check_average_parameter(targets, predictions, average)
        f1 = skl_metrics.f1_score(targets, predictions, average=average,
                                  pos_label=pos_label)

        return f1

    # TODO: what type are the raw predicted values that come out??
    @staticmethod
    def get_auc(targets, raw_predictions, num_classes, average=None,
                pos_label=1):
        """
        Calculate the AUC. Note: raw_predictions and num_classes are required.

        Parameters:
            targets: numpy.ndarray of integers
                The target values.
            raw_predictions: numpy.ndarray of floats
                The raw predicted values, not converted to classes.
            num_classes: int
                The number of classes that will be predicted.
            average: string
                [None, ‘binary’ (def), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
                This parameter is required for multiclass/multilabel targets.
                If None, the scores for each class are returned.
                Otherwise, this determines the type of averaging performed on
                the data. See Scikit learn.
            pos_label: int default 1
                The index value that is positive when in the binary case

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
        elif average == "binary":
            all_class_auc = all_class_auc[pos_label]
        elif average:
            raise NotImplementedError

        return all_class_auc

    @staticmethod
    def get_mse(targets, raw_predictions):
        """
        Calculate the negative MSE.

        Note: raw_predictions and num_classes are required. Negative
        values are returned so the value can be optimized.

        Parameters:
            targets: numpy.ndarray of integers
                The target values.
            raw_predictions: numpy.ndarray of floats
                The raw predicted values, not converted to classes.

        Returns:
              results: float
                The MSE value

        """
        return skl_metrics.mean_squared_error(targets, raw_predictions)

    @staticmethod
    def run_test(network, data_loader, plot=False, save_path=None,
                 pos_label=1, transform_callable=None, **kwargs):
        """
        Will conduct the test suite to determine network strength.

        Calls either _run_test_single_continuous or _run_test_multi
        depending on the number of classes. _run_test_multi
        only works with raw one-hot encoded output values. Note that multi
        class multi outputs are not supported: values will either be converted
        to class values from one-hot encoding or be a single continuous value.
        However, values can be transformed after the forward pass using
        transform_callable.

        Parameters:
            network: nn.Module
                The network
            data_loader : DataLoader
                A DataLoader object to run the test with.
            save_path : string
                Folder to place images in.
            plot: bool
                Determine if graphs should be plotted in real time.
            pos_label: int
                The label that is positive in the binary case for macro
                calculations.
            transform_callable: callable
                A torch function. e.g. torch.round()
            kwargs: dict of keyworded parameters
                Values passed to transform callable (function parameters)

        Returns:
            results : dict

        """
        num_classes = network.num_classes

        if num_classes is None or num_classes == 0:
            raise ValueError('There\'s no classification layer')
        elif num_classes == 1:
            results_dict = \
                Metrics._run_test_single_continuous(
                    network,
                    data_loader,
                    transform_callable=transform_callable,
                    **kwargs)
        else:
            results_dict = Metrics._run_test_multi(network, data_loader,
                                                   plot=plot,
                                                   save_path=save_path,
                                                   pos_label=pos_label)

        return results_dict

    @staticmethod
    def _run_test_single_continuous(network, data_loader,
                                    transform_callable, **kwargs):
        """
        Will conduct the test suite to determine network strength.

        Parameters:
            network: nn.Module
                The network
            data_loader : DataLoader
                A DataLoader object to run the test with.
            transform_callable: callable
                A torch function. e.g. torch.round()

        Returns:
            results : dict

        """
        targets = np.array([v[1] for v in data_loader.dataset])

        predictions = network.forward_pass(
            data_loader=data_loader,
            convert_to_class=False,
            transform_callable=transform_callable,
            **kwargs)

        mse = Metrics.get_mse(targets, predictions)

        logger.info('{} test\'s results'.format(network.name))

        logger.info('\nMean Squared Error: {}'.format(mse))

        return {"mse": mse}

    @staticmethod
    def _run_test_multi(network, data_loader, plot=False, save_path=None,
                        pos_label=1, transform_callable=None):
        """
        Will conduct the test suite to determine network strength.

        No transforms will be conducted on the outputs, save where
        necessary for calculating metric values. You can transform values
        after the forward pass by using transform_callable.

        Parameters:
            network: nn.Module
                The network
            data_loader : DataLoader
                A DataLoader object to run the test with.
            save_path : string
                Folder to place images in.
            plot: bool
                Determine if graphs should be plotted in real time.
            transform_callable: callable
                A torch function. e.g. torch.round()

        Returns:
            results : dict

        """
        num_classes = network.num_classes

        if num_classes > 2:
            average = "macro"
        elif num_classes == 2:
            logger.warning("Will report scores only for pos_label, which is \
                           set to {}".format(pos_label))
            average = "binary"
        else:
            raise ValueError("Incorrect number of classes for run_test_multi."
                             "Need to have more than one class.")

        if plot:
            logger.setLevel(logging.INFO)

        # getting just the y values out of the dataset
        # TODO: store in tensor for continuity?
        targets = np.array([v[1] for v in data_loader.dataset])

        raw_predictions = network.forward_pass(
            data_loader=data_loader,
            transform_callable=transform_callable
        )

        predictions = Metrics.transform_outputs(raw_predictions)

        cm = skl_metrics.confusion_matrix(targets, predictions)
        if plot:
            display_confusion_matrix(cm, save_path=save_path)

        tp, tn, fp, fn = Metrics.get_confusion_matrix_values(targets,
                                                             predictions)

        sensitivity = Metrics.get_sensitivity(targets, predictions)
        sensitivity_macro = Metrics.get_sensitivity(targets, predictions,
                                                    average=average)

        specificity = Metrics.get_specificity(targets, predictions)
        specificity_macro = Metrics.get_specificity(targets, predictions,
                                                    average=average)

        dice = Metrics.get_dice(targets, predictions)
        dice_macro = Metrics.get_dice(targets, predictions, average=average)

        ppv = Metrics.get_ppv(targets, predictions)
        ppv_macro = Metrics.get_ppv(targets, predictions, average=average)

        npv = Metrics.get_npv(targets, predictions)
        npv_macro = Metrics.get_npv(targets, predictions, average=average)

        accuracy = Metrics.get_accuracy(targets, predictions)

        f1 = Metrics.get_f1(targets, predictions)
        f1_macro = Metrics.get_f1(targets, predictions, average=average)

        auc = Metrics.get_auc(targets, raw_predictions, num_classes)
        auc_macro = Metrics.get_auc(targets, raw_predictions, num_classes,
                                    average=average)

        logger.info('{} test\'s results'.format(network.name))

        logger.info('TP: {}'.format(tp))
        logger.info('FP: {}'.format(fp))
        logger.info('TN: {}'.format(tn))
        logger.info('FN: {}'.format(fn))

        logger.info('\nAccuracy: {}'.format(accuracy))

        logger.info('Sensitivity: {}'.format(round_list(sensitivity,
                                                        decimals=3)))
        # accuracy and macro sensitivity are the same when classes are balanced
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

    # TODO: Evaluate these as static methods...
    # noinspection PyUnusedLocal
    @staticmethod
    def bootfold_p_estimate(network, data_loader, n_samples, k, epochs,
                            index_to_iter, ls_feat_vals, retain_graph=None,
                            valid_interv=4, plot=False,
                            save_path=None, **kwargs):
        """
        Performs bootfold - estimation to identify whether training model
        provides statistically significant
        difference in predicting various values for a given feature when
        predicting outcome.

        Parameters:
            network : BaseNetwork
                Network descendant of BaseNetwork.
            data_loader : torch.utils.data.DataLoader
                The DataLoader object containing the totality of the data to
                use for k-fold cross validation.
            n_samples : int
                number of times to randomly sample w/ replacement the
                data_loader and perform boot_cv
            k : int
                The number of folds to split the training into.
            epochs : int
                The number of epochs to train the network per fold.
            index_to_iter : string
                Name of feature within data_loader who's values will be
                iterated to assess difference
            ls_feat_vals : list
                List of values for feature provided in index_to_iter
            retain_graph : {None, boolean}
                Whether retain_graph will be true when .backwards is called.
            valid_interv : int
                Specifies after how many epochs validation should occur.
            plot : boolean
                Whether or not to plot all results in prompt and charts.
            save_path : str
                Where to save all figures and results.
            kwargs: dict of keyworded parameters
                Values passed to transform callable (function parameters)

        Returns:
            p_value : float
        """
        ls_imprv_scores = []

        rand_sample = data.RandomSampler(data_loader.dataset, replacement=True)
        data_loader_args = inspect.signature(data.DataLoader.__init__)
        new_params = {}

        for param in data_loader_args.parameters:
            if param == "self":
                continue
            elif param == "shuffle":
                continue
            elif param == "sampler":
                new_params["sampler"] = rand_sample
            elif param == "batch_sampler":
                new_params["batch_sampler"] = None
            else:
                new_params[param] = getattr(data_loader, param)

        dl_sample = data.DataLoader(**new_params)

        for samp in range(n_samples):
            imprv_score = Metrics._boot_cv(network, dl_sample, k, epochs,
                                           retain_graph, valid_interv, plot,
                                           save_path, index_to_iter,
                                           ls_feat_vals)
            ls_imprv_scores.append(imprv_score)

        tot_num_imprv = float(len(ls_imprv_scores))
        score_below_zero = sum(val <= 0.0 for val in ls_imprv_scores)
        # calculate p value based on change of not improving
        p_val = float(score_below_zero)/float(tot_num_imprv)
        logger.info("Improvement scores: {}"
                    .format(', '.join(map(str, ls_imprv_scores))))
        logger.info("P value for bootfold p estimate: %f.", p_val)

    def _boot_cv(network, data_loader, k, epochs, retain_graph, valid_interv,
                 save_path, plot, index_to_iter, ls_feat_vals):
        """
        Perform a custom cross validation for bootstrapped p estimation.

        Parameters:
            network : BaseNetwork
                Network descendant of BaseNetwork.
            data_loader : torch.utils.data.DataLoader
                The DataLoader object containing the totality of the data to
                use for k-fold cross validation.
            k : int
                The number of folds to split the training into.
            epochs : int
                The number of epochs to train the network per fold.
            retain_graph : {None, boolean}
                Whether retain_graph will be true when .backwards is called.
            valid_interv : int
                Specifies after how many epochs validation should occur.
            plot : boolean
                Whether or not to plot all results in prompt and charts.
            index_to_iter : string
                Index of feature within data_loader who's values will be
                iterated to assess difference
            ls_feat_vals : list
                List of values for feature provided in index_to_iter

        Returns:
            improvement_score : float
        """

        dct_filtered_subj = defaultdict(list)
        ls_filtered_probs = []
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
                dct_scores = _get_probs(network, val_loader, index_to_iter,
                                        ls_feat_vals)
                dct_filtered = _filter_matched_subj(dct_scores, val_loader,
                                                    index_to_iter)
                for ind in dct_filtered:
                    dct_filtered_subj[ind].append(dct_filtered[ind])
                    ls_filtered_probs.append(dct_filtered[ind])
            v_p = np.array(ls_filtered_probs).mean()
            ls_pos_rate = [subj for subj in data_loader.dataset if
                           subj[1].item() == 1]
            v_t = float(len(ls_pos_rate)/len(data_loader.dataset)) * 100
            improvement_score = float(v_p/v_t)
            if np.isnan(improvement_score):
                improvement_score = 0.0
            return improvement_score
        except KeyboardInterrupt:
            logger.info(
                "\n\n***KeyboardInterrupt: Cross validate stopped \
                prematurely.***\n\n")

    @staticmethod
    def stratified_split(dataset, k, strata_column="class_label"):
        """ Split data into k subsets evenly over the given strata_column
        Does not work for greater than 2-D data.

        Parameters:
            dataset : data.dataset
                The dataset to be split.
            k : int
                The number of folds to split the training into.
            strata_column: string or int
                Either "class_label" or integer index of column.
                Default "class_label"

        Returns:
            split_datasets : list
            A list of dataset objects split into k parts.

        """

        # the target column exists in the second tensor
        # and the training data in the first tensor.
        if strata_column == "class_label":
            sr = pd.Series(dataset.tensors[1].numpy())
        else:
            # just let it fail if it doesn't work
            # does not work for complex multidimensional data
            sr = pd.Series(dataset.tensors[0][strata_column].numpy())

        ls = [[] for i in range(k)]

        grps = sr.groupby(by=sr)

        for key, grp in grps:
            inds = np.array(grp.index)
            np.random.shuffle(inds)
            chunks = np.array_split(inds, k)
            # also shuffle which chunks get less items.
            np.random.shuffle(chunks)
            for i in range(k):
                ls[i].extend(chunks[i])

        datasets = [data.Subset(dataset, l) for l in ls]

        return datasets

    @staticmethod
    def cross_validate(network, data_loader, k, epochs,
                       average_results=True, retain_graph=None,
                       valid_interv=4, plot=False, save_path=None,
                       transform_callable=None,
                       stratified=False,
                       strata_column="class_label",
                       **kwargs):
        """
        Perform k-fold cross validation given a Network and DataLoader object.

        Parameters:
            network : BaseNetwork
                Network descendant of BaseNetwork.
            data_loader : torch.utils.data.DataLoader
                The DataLoader object containing the totality of the data to
                use for k-fold cross validation.
            k : int
                The number of folds to split the training into.
            epochs : int
                The number of epochs to train the network per fold.
            average_results : boolean
                Whether or not to return results from all folds or just an
                average.
            retain_graph : {None, boolean}
                Whether retain_graph will be true when .backwards is called.
            valid_interv : int
                Specifies after how many epochs validation should occur.
            plot : boolean
                Whether or not to plot all results in prompt and charts.
            save_path : str
                Where to save all figures and results.
            transform_callable: callable
                A torch function. e.g. torch.round()
            stratified: bool
                Whether to stratify the dataset. Default false.
            strata_column: string or int
                Either "class_label" or integer index of column.
                Default "class_label"
            kwargs: dict of keyworded parameters
                Values passed to transform callable (function parameters)


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

        if stratified:
            dataset_splits = Metrics.stratified_split(data_loader.dataset,
                                                      k, strata_column)
        else:
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
                    save_path=save_path, plot=plot,
                    transform_callable=transform_callable,
                    **kwargs)

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
            logger.info(averaged_all_results)
            return averaged_all_results

        else:
            logger.info(all_results)
            return all_results

    @staticmethod
    def conduct_sensitivity_analysis(network, data_loader, filename,
                                     features=None, cutoff=20):
        """
        Will conduct tests to figure out directionality of features by finding all the unique feature values present in
        the dataset and setting the feature for all examples to every unique value of that feature.

        Parameters
        ----------
            network : BaseNetwork
                Network descendant of BaseNetwork.
            data_loader : DataLoader
                A DataLoader object which contains all the data
            filename: string
                The name of the file to save the test results to
            features: list
                Feature names ordered as the data in the data_loader
            cutoff: int
                Maximum number of unique feature values for a particular
                feature that will be tested

        Returns
        -------
        dataframe characterizing the number of examples classified in the
        different classes when altering the input to constant feature values
        """

        if network._num_classes is None or network._num_classes == 0:
            raise ValueError('There\'s no classification layer')

        if features is None:
            features = [i for i in range(data_loader.dataset[:][0].shape[1])]

        test_df = None
        col_headers = None

        for feature_idx in range(len(features)):
            feature_val_list = data_loader.dataset[:][0][
                               :, feature_idx].tolist()

            unique_feature_values = list(set(feature_val_list))

            if len(unique_feature_values) > cutoff:
                logger.info("================================")
                logger.info("Cutting off number of features to be tested for "
                            "feature {} from {} to {}".format(
                    features[feature_idx], len(unique_feature_values), cutoff))
                logger.info("================================")

                # unique_feature_values are sorted, this next line
                # pulls cutoff num of features evenly spaced from the list of
                # unique features since the list contains more unique features
                # than the cutoff
                unique_feature_values = unique_feature_values[0::math.ceil(
                        len(unique_feature_values) / cutoff)]
            data_loader_temp = copy.deepcopy(data_loader)
            for unique_feature_idx in range(len(unique_feature_values)):
                data_loader_temp.dataset[:][0][:, feature_idx] = \
                    unique_feature_values[unique_feature_idx]
                logger.info("Testing value {} for feature {}".format(
                    unique_feature_values[unique_feature_idx],
                    features[feature_idx]))

                targets = data_loader.dataset[:][1].tolist()

                raw_predictions = network.forward_pass(
                    data_loader=data_loader_temp)

                predictions = Metrics.transform_outputs(raw_predictions)

                tp, _, fp, _ = Metrics.get_confusion_matrix_values(targets,
                                                                   predictions)

                # print(raw_predictions)
                # print(predictions)

                if test_df is None and col_headers is None:
                    col_headers = ["Feature", "Value"]
                    for i in range(network._num_classes):
                        col_headers.append("Number of examples classified as "
                                           "class " + str(i))
                    test_df = pd.DataFrame(columns=col_headers)

                row_to_append = [features[feature_idx],
                                 unique_feature_values[unique_feature_idx]]

                row_to_append = row_to_append + [tp_i + fp_i for (tp_i, fp_i)
                                                 in zip(tp, fp)]

                df = pd.DataFrame(np.array(row_to_append).reshape(1, -1),
                                  columns=col_headers)
                test_df = test_df.append(df)
            logger.info("***************************")
        test_df.to_csv("{}.csv".format(filename), index=False)
        return test_df

