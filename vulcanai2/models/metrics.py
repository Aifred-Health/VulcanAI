# coding=utf-8
"""Defines the network test suite."""
import torch
from torch.utils import data

import math
import numpy as np
from sklearn import metrics as skl_metrics

from .utils import get_confusion_matrix, round_list
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
    # TODO: why does use_unlabeled exist?
    #def __init__(self, use_unlabeled=False):
    def __init__(self):
        """Initialize the metrics class for a BaseNetwork."""
        #self.num_class = num_class
        # self.mat = np.zeros((self.num_class, self.num_class), dtype=np.float)
        #self.list_classes = list(range(self.num_class))

    # def update(self, predictions, targets):
    #     if not(isinstance(predictions, np.ndarray)) or not(isinstance(targets, np.ndarray)):
    #         print("Expected ndarray")

    #     elif len(targets.shape) == 3:        # batched spatial target
    #         if len(predictions.shape) == 4:  # prediction is 1 hot encoded
    #             temp_predictions = np.argmax(predictions, axis=1).flatten()
    #         elif len(predictions.shape) == 3:
    #             temp_predictions = predictions.flatten()
    #         else:
    #             print("Predictions and Targets does not match")
    #         temp_targets = targets.flatten()

    #     elif len(targets.shape) == 2:        # spatial target
    #         if len(predictions.shape) == 3:  # prediction is 1 hot encoded
    #             temp_predictions = np.argmax(predictions, axis=1).flatten()
    #         elif len(predictions.shape) == 2:
    #             temp_predictions = predictions.flatten()
    #         else:
    #             print("Predictions and Targets does not match")
    #         temp_targets = targets.flatten()

    #     elif len(targets.shape) == 1:
    #         if len(predictions.shape) == 2:  # prediction is 1 hot encoded
    #             temp_predictions = np.argmax(predictions, axis=1).flatten()
    #         elif len(predictions.shape) == 1:
    #             temp_predictions = predictions
    #         else:
    #             print("Predictions and Targets does not match")
    #         temp_targets = targets
    #     else:
    #         print("Data with this dimension cannot be handled")

    #     self.mat += confusion_matrix(temp_targets, temp_predictions, labels=self.list_classes)

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

    # TODO: class # should correspond with self.num_class
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

    # TODO: check types
    def get_confusion_matrix_values(self, confusion_matrix):
        """
        Will calculate the tp, tn, fp, fn values given a confusion matrix
        Parameters
        ----------
        confusion_matrix: sklearn.metrics.confusion_matrix
        The confusion matrix

        Returns
        -------
        tp, tn, fp, fn:  np.float32
        The values calculated
        """
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

    def get_sensitivity(self, tp, fn):
        """
        Calculate the sensitivity
        Parameters
        ----------
        tp: array of np.float32
        true positives
        fn: array of np.float32
        false negatives

        Returns
        -------
        sensitivity: array of np.float32
        The sensitivity
        """
        sensitivity = np.nan_to_num(tp / (tp + fn))  # recall
        return sensitivity

    def get_specificity(self, tn, fp):
        """
        Calculate the specificity
        Parameters
        ----------
        tn: array of np.float32
        true negatives
        fp: array of np.float32
        false negatives

        Returns
        -------
        specificity: array of np.float32
        The specificity
        """

        specificity = np.nan_to_num(tn / (tn + fp))
        return specificity

    def get_dice(self, tp, fp, fn):
        """
        Calculates the dice metric
        Parameters
        ----------
        tp: array of np.float32
        true positives
        fp: array of np.float32
        false positives
        fn: array of np.float32
        false negatives

        Returns
        -------
        dice: array of np.float32
        The dice metric.
        """

        dice = 2 * tp / (2 * tp + fp + fn)
        return dice

    def get_ppv(self, tp, fp):
        """
        Calculate the positive predictive value
        Parameters
        ----------
        tp: array of np.float32
        true positives
        fp: array of np.float32
        false positives

        Returns
        -------
        ppv: array of np.float32
        the positive predictive value
        """

        ppv = np.nan_to_num(tp / (tp + fp))
        return ppv

    def get_npv(self, tn, fn):
        """
        Calculates the negative predictive value
        Parameters
        ----------
        tn: array of np.float32
        the true negatives
        fn: array of np.float32
        the false negatives

        Returns
        -------
        npv: array of np.float32
        The negative predictive value
        """

        npv = np.nan_to_num(tn / (tn + fn))
        return npv

    def get_accuracy(self, tp, confusion_matrix):
        """
        Calculate the accuracy
        Parameters
        ----------
        tp: array of np.float32:
        true positives
        confusion_matrix: sklearn.metrics.confusion_matrix
        The confusion matrix

        Returns
        -------
        accuracy: array of np.float32
        The accuracy

        """

        accuracy = np.sum(tp) / np.sum(confusion_matrix)
        return accuracy

    def get_f1(self, ppv, sensitivity):
        """
        Calculate the f1 score
        Parameters
        ----------
        ppv: array of np.float32
        positive predictive value
        sensitivity: array of np.float32
        sensitivity

        Returns
        -------
        f1: array of np.float32
        the f1 score
        """

        f1 = np.nan_to_num(2 * (ppv * sensitivity) / (ppv + sensitivity))
        return f1

    def get_f1_macro(self, ppv, sensitivity):
        """
        Calculate the average f1 score
        Parameters
        ----------
        ppv: array of np.float32
        positive predictive value
        sensitivity: array of np.float32
        sensitivity

        Returns
        -------
        f1_macro: np.float32
        """

        f1_macro = np.average(np.nan_to_num(2 * sensitivity * ppv / (sensitivity + ppv)))
        return f1_macro

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
        test_y = np.array([v[1] for v in data_loader.dataset]) #TODO: store in tensor for continuity?

        raw_prediction = network.forward_pass(
            data_loader=data_loader,
            convert_to_class=False)

        class_prediction = self.extract_class_labels(raw_prediction)

        confusion_matrix = get_confusion_matrix(
            predictions=class_prediction,
            targets=test_y
        )

        if plot:
            display_confusion_matrix(confusion_matrix)

        tp, tn, fp, fn = self.get_confusion_matrix_values(confusion_matrix)

        sensitivity = self.get_specificity(tn, fp)
        sensitivity_macro = np.average(sensitivity)

        specificity = self.get_specificity(tn, fp)
        specificity_macro = np.average(specificity)

        dice = self.get_dice(tp, fp, fn)
        dice_macro = np.average(dice)

        ppv = self.get_ppv(tp, fp)
        ppv_macro = np.average(ppv)

        npv = self.get_npv(tn, fn)
        npv_macro = np.average(npv)

        accuracy = self.get_accuracy(tp, confusion_matrix)

        f1 = self.get_f1(self, ppv, sensitivity)
        f1_macro = self.get_f1_macro(ppv, sensitivity)

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

        all_class_auc = []
        for i in range(network._num_classes):
            if network._num_classes == 1:
                fpr, tpr, _ = skl_metrics.roc_curve(test_y,
                                                    raw_prediction,
                                                    pos_label=1)
            else:
                fpr, tpr, _ = skl_metrics.roc_curve(test_y,
                                                    raw_prediction[:, i],
                                                    pos_label=i)

            auc = skl_metrics.auc(fpr, tpr)
            all_class_auc += [auc]
        all_class_auc_macro = np.average(all_class_auc)

        return {
            'accuracy': float(accuracy),
            'macro_sensitivity': float(sensitivity_macro),
            'macro_specificity': float(specificity_macro),
            'avg_dice': float(dice_macro),
            'macro_ppv': float(ppv_macro),
            'macro_npv': float(npv_macro),
            'macro_f1': float(f1_macro),
            'macro_auc': float(np.average(all_class_auc))
        }

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
