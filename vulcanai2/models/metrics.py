# coding=utf-8
"""Defines the network test suite."""
import torch
from torch.utils import data

import math
import numpy as np
from sklearn import metrics as skl_metrics

from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score

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

    Parameters
    ----------
    """
    def __init__(self):
        """Initialize the metrics class for a BaseNetwork."""

    def get_score(self, predictions, targets, metric='accuracy'):
        """
        Calculate some defined score given predictions and targets.

        Parameters
        ----------
        predictions : torch.Tensor
            Network output of shape [batch, num_classes].
        targets : torch.LongTensor
            The truth values of shape [batch].
        metric : str
            The metric to calculate and return.

        Returns
        -------
        score : float
            The specified metric to calculate.

        """
        if metric == 'accuracy':
            # TODO: Use extract_class_labels
            max_index = predictions.max(dim=1)[1]
            correct = (max_index == targets).sum() # TODO: this doesn't seem correct
            accuracy = int(correct.data) / len(targets)
            return accuracy
        else:
            raise NotImplementedError(
                'Metric {} not available.'.format(metric))

    # noinspection PyMethodMayBeStatic
    def extract_class_labels(self, in_matrix):
        """
        Reformat truth matrix to be the classes in a 1D array.

        Parameters
        ----------
        in_matrix : numpy.ndarray or torch.Tensor
            One-hot matrix of shape [batch, num_classes].

        Returns
        -------
        class_list : numpy.ndarray
            1D class array.

        """
        if isinstance(in_matrix, torch.Tensor):
            in_matrix = in_matrix.detach().numpy()
        # For one-hot encoded entries
        if in_matrix.shape[1] > 1:
            return np.argmax(in_matrix, axis=1)
        # For binary entries
        elif in_matrix.shape[1] == 1:
            return np.around(in_matrix)

    @staticmethod
    def get_confusion_matrix_values(targets, predictions):
        """
        Will calculate the tp, tn, fp, fn values given a confusion matrix
        Parameters
        ----------
        predictions: numpy.ndarray of integers
        the predicted values

        targets: numpy.ndarray of integers
        the target values

        Returns
        -------
        tp, tn, fp, fn:  integer
        The values calculated
        """

        tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()

        return tp, tn, fp, fn

    @staticmethod
    def _check_average_parameters(targets, predictions, average=None):
        """
        Checks to see if average parameter is suitable for the data. Throws ValueError.
        Parameters
        ----------
        predictions: numpy.ndarray of integers
        the predicted values

        targets: numpy.ndarray of integers
        the target values

        average: string, [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
        This parameter is required for multiclass/multilabel targets.
        If None, the scores for each class are returned.
        Otherwise, this determines the type of averaging performed on the data:
        See scikit learn

        Returns
        -------
        True

        """

        if not average:
            return True

        if np.unique(predictions).size <= 2 and np.unique(targets).size <= 2:
            if "binary" not in average:
                raise ValueError(
                    "You must provide binary as the average function if binary data"
                )

        if np.unique(predictions).size >= 2 or np.unique(targets).size >= 2:
            if "binary" in average:
                raise ValueError(
                    "You cannot provide binary as the average function if non binary data"
                )

        return True

    @staticmethod
    def get_sensitivity(targets, predictions, average=None):
        """
        Calculate the sensitivity
        Parameters
        ----------
        predictions: numpy.ndarray of integers
        the predicted values

        targets: numpy.ndarray of integers
        the target values

        average: string, [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
        This parameter is required for multiclass/multilabel targets.
        If None, the scores for each class are returned.
        Otherwise, this determines the type of averaging performed on the data.
        See scikit learn

        Returns
        -------
        sensitivity: np.float32 or array of np.float32
        The sensitivity
        """

        Metrics._check_average_parameters(targets, predictions, average)
        sensitivity = recall_score(targets, predictions, average)

        return sensitivity

    @staticmethod
    def get_specificity(targets, predictions, average=None):
        """
        Calculate the specificity
        Parameters
        ----------
        predictions: numpy.ndarray of integers
        the predicted values

        targets: numpy.ndarray of integers
        the target values

        average: string, [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
        This parameter is required for multiclass/multilabel targets.
        If None, the scores for each class are returned.
        Otherwise, this determines the type of averaging performed on the data.
        See scikit learn
        Returns
        -------
        specificity: np.float32 or array of np.float32
        The specificity
        """

        Metrics._check_average_parameters(targets, predictions, average)
        specificity = precision_score(targets, predictions, average)

        return specificity

    @staticmethod
    def get_dice(targets, predictions, average=None):
        """
        Calculates the dice metric
        Parameters
        ----------
        predictions: numpy.ndarray of integers
        the predicted values

        targets: numpy.ndarray of integers
        the target values

        average: string, [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
        Only None and 'macro' currently implemented.
        This parameter is required for multiclass/multilabel targets.
        If None, the scores for each class are returned.
        Otherwise, this determines the type of averaging performed on the data.
        See scikit learn

        Returns
        -------
        dice: np.float32 or array of np.float32
        The dice metric.
        """

        tp, _, fp, fn = Metrics.get_confusion_matrix_values(targets, predictions)
        dice = 2 * tp / (2 * tp + fp + fn)

        # TODO: implement other options
        if "macro" in average:
            dice = np.average(dice)
        elif average:
            raise NotImplementedError

        return dice

    @staticmethod
    def get_ppv(targets, predictions, average=None):
        """
        Calculate the positive predictive value
        Parameters
        ----------
        predictions: numpy.ndarray of integers
        the predicted values

        targets: numpy.ndarray of integers
        the target values

        average: string, [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
        Only None and 'macro' currently implemented.
        This parameter is required for multiclass/multilabel targets.
        If None, the scores for each class are returned.
        Otherwise, this determines the type of averaging performed on the data.
        See scikit learn

        Returns
        -------
        ppv: float32 or array of np.float32
        the positive predictive value
        """
        tp, _, fp, _ = Metrics.get_confusion_matrix_values(targets, predictions)
        ppv = np.nan_to_num(tp / (tp + fp))

        # TODO: implement other options
        if "macro" in average:
            ppv = np.average(ppv)
        elif average:
            raise NotImplementedError

        return ppv

    @staticmethod
    def get_npv(targets, predictions, average=None):
        """
        Calculates the negative predictive value
        Parameters
        ----------
        predictions: numpy.ndarray of integers
        the predicted values

        targets: numpy.ndarray of integers
        the target values

        average: string, [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
        Only None and 'macro' currently implemented.
        This parameter is required for multiclass/multilabel targets.
        If None, the scores for each class are returned.
        Otherwise, this determines the type of averaging performed on the data.
        See scikit learn

        Returns
        -------
        npv: np.float32 or array of np.float32
        The negative predictive value
        """
        _ , tn, _, fn = Metrics.get_confusion_matrix_values(targets, predictions)

        npv = np.nan_to_num(tn / (tn + fn))

        # TODO: implement other options
        if "macro" in average:
            npv = np.average(npv)
        elif average:
            raise NotImplementedError

        return npv

    @staticmethod
    def get_accuracy(targets, predictions):
        """
        Calculate the accuracy
        Parameters
        ----------
       predictions: numpy.ndarray of integers
        the predicted values

        targets: numpy.ndarray of integers
        the target values


        Returns
        -------
        accuracy: array of np.float32
        The accuracy

        """
        accuracy = accuracy_score(predictions, targets)

        return accuracy

    @staticmethod
    def get_f1(targets, predictions, average=None):
        """
        Calculate the f1 score
        Parameters
        ----------
        predictions: numpy.ndarray of integers
        the predicted values

        targets: numpy.ndarray of integers

        average: string, [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
        This parameter is required for multiclass/multilabel targets.
        If None, the scores for each class are returned.
        Otherwise, this determines the type of averaging performed on the data:
        See scikit learn

        Returns
        -------
        f1: np.float32 or array of np.float32
        The f1 score
        """

        Metrics._check_average_parameters(targets, predictions, average)
        f1 = f1_score(targets, predictions, average)
        return f1

    @staticmethod
    def get_auc(targets, raw_predictions, num_classes, average=None):
        """
        Calculate the auc
        Parameters
        ----------
        raw_predictions: numpy.ndarray of integers
        the raw predicted values, not converted to classes.

        targets: numpy.ndarray of integers

        average: string, [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
        This parameter is required for multiclass/multilabel targets.
        If None, the scores for each class are returned.
        Otherwise, this determines the type of averaging performed on the data:
        See scikit learn

        Returns
        -------
        f1: np.float32 or array of np.float32
        The auc
        """

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
        if "macro" in average:
            all_class_auc = np.average(all_class_auc)
        elif average:
            raise NotImplementedError

        return all_class_auc

    def run_test(self, network, data_loader, figure_path=None, plot=False):
        """
        Will conduct the test suite to determine network strength.

        Parameters
        ----------
        data_loader : DataLoader
            A DataLoader object to run the test with.
        figure_path : string
            Folder to place images in.
        plot: bool
            Determine if graphs should be plotted in real time.

        Returns
        -------
        results : dict

        """
        if plot:
            logger.setLevel(logging.INFO)

        if network._num_classes is None or \
           network._num_classes == 0:
            raise ValueError('There\'s no classification layer')

        # getting just the y values out of the dataset
        targets = np.array([v[1] for v in data_loader.dataset]) #TODO: store in tensor for continuity?

        raw_predictions = network.forward_pass(
            data_loader=data_loader,
            convert_to_class=False)

        predictions = self.extract_class_labels(raw_predictions)

        #confusion matrix is from sklearn namespace
        cm = confusion_matrix(targets, predictions) #TODO: you made them backwards
        if plot:
            display_confusion_matrix(cm)

        tp, tn, fp, fn = Metrics.get_confusion_matrix_values(targets, predictions)

        sensitivity = Metrics.get_specificity(targets, predictions)
        sensitivity_macro = Metrics.get_specificity(targets, predictions, average="macro")

        specificity = Metrics.get_specificity(targets, predictions)
        specificity_macro = Metrics.get_specificity(targets, predictions, average="macro")

        dice = Metrics.get_dice(targets, predictions)
        dice_macro = Metrics.get_dice(targets, predictions, average="macro")

        ppv = Metrics.get_ppv(targets, predictions)
        ppv_macro = Metrics.get_ppv(targets, predictions, average="macro")

        npv = Metrics.get_npv(targets, predictions)
        npv_macro = Metrics.get_npv(targets, predictions, average="macro")

        accuracy = Metrics.get_accuracy(targets, predictions)

        f1 = Metrics.get_f1(targets, predictions)
        f1_macro = Metrics.get_f1(targets, predictions, average="macro")

        auc = Metrics.get_auc(targets, predictions)
        auc_macro = Metrics.get_auc(targets, predictions, average="macro")

        logger.info('{} test\'s results'.format(network.name))

        logger.info('TP: {}'.format(tp))
        logger.info('FP: {}'.format(fp))
        logger.info('TN: {}'.format(tn))
        logger.info('FN: {}'.format(fn))

        logger.info('\nAccuracy: {}'.format(accuracy))

        logger.info('Sensitivity: {}'.format(round_list(sensitivity, decimals=3)))
        logger.info('\tMacro Sensitivity: {:.4f}'.format(sensitivity_macro))

        logger.info('Specificity: {}'.format(round_list(specificity, decimals=3)))
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


    #TODO: include support

    def cross_validate(self, network, data_loader, k, epochs,
                       average_results=True, retain_graph=None,
                       valid_interv=4, plot=False, figure_path=None):
        """
        Perform k-fold cross validation given a Network and DataLoader object.

        Parameters
        ----------
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
        figure_path : str
            Where to save all figures and results.

        Returns
        -------
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
                    valid_interv=valid_interv, plot=plot)
                # Validate network performance on validation data loader.
                results = self.run_test(
                    cross_val_network, val_loader,
                    figure_path=figure_path, plot=plot)

                logger.info(results)
                for m in results:
                    all_results[m].append(results[m])

        # TODO: we could show something better here like calculate
        # all the results so far
        except KeyboardInterrupt:
            print("\n\n***KeyboardInterrupt: Cross validate stopped prematurely.***\n\n")

        if average_results:
            averaged_all_results = {}
            for m in all_results:
                averaged_all_results[m] = np.mean(all_results[m])
            return averaged_all_results
        else:
            return all_results
