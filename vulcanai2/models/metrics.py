"""Defines the network test suite."""
import numpy as np

import torch
import torch.nn.functional as F

import math
import numpy as np
from sklearn import metrics as skl_metrics

from .utils import get_confusion_matrix, round_list
from ..plotters.visualization import display_confusion_matrix

from copy import deepcopy
import datetime

from collections import Counter

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


class Metrics(object):

    def __init__(self, num_class, use_unlabeled=False):
        self.num_class = num_class
        # self.mat = np.zeros((self.num_class, self.num_class), dtype=np.float)
        self.list_classes = list(range(self.num_class))
  
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
        if metric == 'accuracy':
            max_index = predictions.max(dim=1)[1]
            correct = (max_index == targets).sum()
            accuracy = int(correct.data) / len(targets)
            return accuracy
        else:
            raise NotImplementedError('Metric not available.')

    # TODO: class # should correspond with self.num_class
    def get_class(self, in_matrix):
        """
        Reformat truth matrix to be the classes in a 1D array.

        :param n_matrix: one-hot matrix
        :return: 1D Class array
        """
        if isinstance(in_matrix, torch.Tensor):
            in_matrix = in_matrix.detach().numpy()
        # For one-hot encoded entries
        if in_matrix.shape[1] > 1:
            return np.argmax(in_matrix, axis=1)
        # For binary entries
        elif in_matrix.shape[1] == 1:
            return np.around(in_matrix)

    # TODO: Modify to use val loader
    def run_test(self, model, data_loader, figure_path=None, plot=False):
        """
        Will conduct the test suite to determine model strength.

        :param data_loader: a DataLoader object to run the test with
        :param figure_path: string, folder to place images in.
        :param plot: bool, determines if graphs should be plotted.
        :return: results dictionary
        """
        if plot:
            logger.setLevel(logging.INFO)

        if model._num_classes is None or \
            model._num_classes == 0 or \
            not hasattr(model, 'network_tail'):
            raise ValueError('There\'s no classification layer')

        test_y = data_loader.dataset.test_labels

        raw_prediction = model.forward_pass(
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
        tn = (np.array([np.sum(confusion_matrix)] *
                    confusion_matrix.shape[0]) -
            confusion_matrix.sum(axis=0) -
            confusion_matrix.sum(axis=1) + tp).astype('float32')
        # sum each column and remove diagonal
        fp = (confusion_matrix.sum(axis=0) - tp).astype('float32')
        # sum each row and remove diagonal
        fn = (confusion_matrix.sum(axis=1) - tp).astype('float32')

        sens = np.nan_to_num(tp / (tp + fn))  # recall
        spec = np.nan_to_num(tn / (tn + fp))
        sens_macro = np.average(sens)  # sens_micro = np.nan_to_num(sum(tp) / (sum(tp) + sum(fn)))
        spec_macro = np.average(spec)  # sens_micro = np.nan_to_num(sum(tn) / (sum(tn) + sum(fp)))
        dice = 2 * tp / (2 * tp + fp + fn)
        ppv = np.nan_to_num(tp / (tp + fp))  # precision
        ppv_macro = np.average(ppv)    # ppv_micro = np.nan_to_num(sum(tp) / (sum(tp) + sum(fp)))
        npv = np.nan_to_num(tn / (tn + fn))
        npv_macro = np.average(npv)    # npv_micro = np.nan_to_num(sum(tn) / (sum(tn) + sum(fn)))
        accuracy = np.sum(tp) / np.sum(confusion_matrix)
        f1 = np.nan_to_num(2 * (ppv * sens) / (ppv + sens))
        f1_macro = np.average(np.nan_to_num(2 * sens * ppv / (sens + ppv)))

        logger.info('{} test\'s results'.format(model.name))

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

        logger.info('Positive Predictive Value: {}'.format(round_list(ppv, decimals=3)))
        logger.info('\tMacro Positive Predictive Value: {:.4f}'.format(ppv_macro))

        logger.info('Negative Predictive Value: {}'.format(round_list(npv, decimals=3)))
        logger.info('\tMacro Negative Predictive Value: {:.4f}'.format(npv_macro))

        logger.info('F1-score: {}'.format(round_list(f1, decimals=3)))
        logger.info('\tMacro f1-score: {:.4f}'.format(f1_macro))

        all_class_auc = []
        for i in range(model._num_classes):
            if model._num_classes == 1:      
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
            'accuracy': accuracy,
            'macro_sensitivity': sens_macro,
            'macro_specificity': spec_macro,
            'avg_dice': np.average(dice),
            'macro_ppv': ppv_macro,
            'macro_npv': npv_macro,
            'macro_f1': f1_macro,
            'macro_auc': np.average(all_class_auc)
        }

    # TODO:  Needs to be updated to use train loader
    def k_fold_validation(self, model, train_x, train_y, k=5, epochs=10,
                          batch_ratio=1.0, plot=False):
        """
        Conduct k fold cross validation on a network.

        Args:
            model: BaseNetwork object you want to cross validate
            train_x: ndarray of shape (batch, features), train samples
            train_y: ndarray of shape(batch, classes), train labels
            k: int, how many folds to run
            batch_ratio: float, 0-1 for % of total to allocate for a batch
            epochs: int, number of epochs to train each fold

        Returns final metric dictionary
        """
        try:
            model.save_name
        except:
            model.save_model()
        chunk_size = int((train_x.shape[0]) / k)
        results = []
        timestamp = "{date:%Y-%m-%d_%H:%M:%S}".format(date=datetime.datetime.now())
        for i in range(k):
            val_x = train_x[i * chunk_size:(i + 1) * chunk_size]
            val_y = train_y[i * chunk_size:(i + 1) * chunk_size]
            tra_x = np.concatenate(
                (train_x[:i * chunk_size], train_x[(i + 1) * chunk_size:]),
                axis=0
            )
            tra_y = np.concatenate(
                (train_y[:i * chunk_size], train_y[(i + 1) * chunk_size:]),
                axis=0
            )
            net = deepcopy(model)
            net.fit(
                epochs=epochs,
                train_x=tra_x,
                train_y=tra_y,
                val_x=val_x,
                val_y=val_y,
                batch_ratio=batch_ratio,
                plot=plot
            )
            results += [Counter(self.run_test(
                net,
                val_x,
                val_y,
                figure_path='figures/kfold_{}_{}'.format(model.name, timestamp),
                plot=plot))]
            del net
        aggregate_results = reduce(lambda x, y: x + y, results)

        print ('\nFinal Cross validated results')
        print ('-----------------------------')
        for metric_key in aggregate_results.keys():
            aggregate_results[metric_key] /= float(k)
            print ('{}: {:.4f}'.format(metric_key, aggregate_results[metric_key]))

        return aggregate_results