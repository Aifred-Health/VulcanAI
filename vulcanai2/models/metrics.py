# coding=utf-8
"""Defines the network test suite."""
import torch

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
            # TODO: Use get_class
            max_index = predictions.max(dim=1)[1]
            correct = (max_index == targets).sum() # TODO: this doesn't seem correct
            accuracy = int(correct.data) / len(targets)
            return accuracy
        else:
            raise NotImplementedError('Metric not available.')

    # TODO: class # should correspond with self.num_class
    # noinspection PyMethodMayBeStatic
    def get_class(self, in_matrix):
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
        test_y = np.array([v[1] for v in data_loader.dataset])

        raw_prediction = network.forward_pass(
            data_loader=data_loader,
            convert_to_class=False)

        class_prediction = self.get_class(raw_prediction)

        confusion_matrix = get_confusion_matrix(
            predictions=class_prediction,
            targets=test_y
        )

        if plot:
            display_confusion_matrix(confusion_matrix)

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

        sens = np.nan_to_num(tp / (tp + fn))  # recall
        spec = np.nan_to_num(tn / (tn + fp))
        sens_macro = np.average(sens)
        # sens_micro = np.nan_to_num(sum(tp) / (sum(tp) + sum(fn)))
        spec_macro = np.average(spec)
        # sens_micro = np.nan_to_num(sum(tn) / (sum(tn) + sum(fp)))
        dice = 2 * tp / (2 * tp + fp + fn)
        ppv = np.nan_to_num(tp / (tp + fp))  # precision
        ppv_macro = np.average(ppv)
        # ppv_micro = np.nan_to_num(sum(tp) / (sum(tp) + sum(fp)))
        npv = np.nan_to_num(tn / (tn + fn))
        npv_macro = np.average(npv)
        # npv_micro = np.nan_to_num(sum(tn) / (sum(tn) + sum(fn)))
        accuracy = np.sum(tp) / np.sum(confusion_matrix)
        f1 = np.nan_to_num(2 * (ppv * sens) / (ppv + sens))
        f1_macro = np.average(np.nan_to_num(2 * sens * ppv / (sens + ppv)))

        logger.info('{} test\'s results'.format(network.name))

        logger.info('TP: {}'.format(tp))
        logger.info('FP: {}'.format(fp))
        logger.info('TN: {}'.format(tn))
        logger.info('FN: {}'.format(fn))

        logger.info('\nAccuracy: {}'.format(accuracy))

        logger.info('Sensitivity: {}'.format(round_list(sens, decimals=3)))
        logger.info('\tMacro Sensitivity: {:.4f}'.format(sens_macro))

        logger.info('Specificity: {}'.format(round_list(spec, decimals=3)))
        logger.info('\tMacro Specificity: {:.4f}'.format(spec_macro))

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

        return {
            'accuracy': float(accuracy),
            'macro_sensitivity': float(sens_macro),
            'macro_specificity': float(spec_macro),
            'avg_dice': float(np.average(dice)),
            'macro_ppv': float(ppv_macro),
            'macro_npv': float(npv_macro),
            'macro_f1': float(f1_macro),
            'macro_auc': float(np.average(all_class_auc))
        }

    def cross_validate(self, network, data_loader, k, epochs, return_average_results=True, retain_graph=None,
                       valid_interv=4, plot=False, figure_path=None):
        """
        Trains the network on the provided data.
        :param data_loader: The DataLoader object containing the training data
        :param k: The number of folds
        :param epochs: The number of epochs per fold
        :param retain_graph: Specifies whether retain_graph will be true when .backwards is called.
        :param valid_interv: Specifies when validation should occur. Not yet implemented.
        :return: None
        """

        from torch.utils.data import TensorDataset

        all_results = defaultdict(lambda: [])

        # TODO: this whole section is really clunky
        # Getting the fold sequence.
        fold_len = math.floor(data_loader.dataset.__len__() / k)
        rem = data_loader.dataset.__len__() % k
        fold_seq = []

        for i in range(k-1):
            fold_seq.append(fold_len)
        if rem == 0:
            fold_seq.append(fold_len)
        else:
            fold_seq.append(fold_len+rem) #last one is the longest if unequal

        dataset_splits = torch.utils.data.random_split(data_loader.dataset, fold_seq)

        batch_size = data_loader.batch_size

        # #TODO: improve the copying of parameters
        if isinstance(data_loader.sampler, torch.utils.data.sampler.RandomSampler):
            shuffle = True
        else:
            shuffle = False

        try:
            for fold in range(k):

                # TODO: this may break on different devices?? test.
                cross_val_network = copy.deepcopy(network)

                # TODO: properly pass params
                train_dataset = torch.utils.data.ConcatDataset(dataset_splits[:fold] + dataset_splits[fold+1:])
                val_dataset = dataset_splits[fold]

                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

                cross_val_network.fit(train_loader, val_loader, epochs,
                                      retain_graph=retain_graph, valid_interv=valid_interv, plot=plot)

                results = self.run_test(cross_val_network, val_loader, figure_path=figure_path, plot=plot)
                for m in results:
                    all_results[m].append(results[m])

        # TODO: we could show something better here like calculate all the results so far
        except KeyboardInterrupt:
            print("\n\n**********KeyboardInterrupt: Training stopped prematurely.**********\n\n")

        if return_average_results:
            averaged_all_results = {}
            for m in all_results:
                averaged_all_results[m] = np.mean(all_results[m])
            return averaged_all_results
        else:
            return all_results
