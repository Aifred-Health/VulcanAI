import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

import math
import numpy as np
from sklearn import metrics as skl_metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from copy import deepcopy
import datetime

from collections import Counter

import logging
logger = logging.getLogger(__name__)

class Metrics(object):

    def __init__(self, num_class, use_unlabeled=False):
        self.num_class = num_class
        self.mat = np.zeros((self.num_class, self.num_class), dtype=np.float)
        self.list_classes = list(range(self.num_class))
  
    def update(self, predictions, targets):
        if not(isinstance(predictions, np.ndarray)) or not(isinstance(targets, np.ndarray)):
            print("Expected ndarray")

        elif len(targets.shape) == 3:        # batched spatial target
            if len(predictions.shape) == 4:  # prediction is 1 hot encoded
                temp_predictions = np.argmax(predictions, axis=1).flatten()
            elif len(predictions.shape) == 3:
                temp_predictions = predictions.flatten()
            else:
                print("Predictions and Targets does not match")
            temp_targets = targets.flatten()

        elif len(targets.shape) == 2:        # spatial target
            if len(predictions.shape) == 3:  # prediction is 1 hot encoded
                temp_predictions = np.argmax(predictions, axis=1).flatten()
            elif len(predictions.shape) == 2:
                temp_predictions = predictions.flatten()
            else:
                print("Predictions and Targets does not match")
            temp_targets = targets.flatten()

        elif len(targets.shape) == 1:
            if len(predictions.shape) == 2:  # prediction is 1 hot encoded
                temp_predictions = np.argmax(predictions, axis=1).flatten()
            elif len(predictions.shape) == 1:
                temp_predictions = predictions
            else:
                print("Predictions and Targets does not match")
            temp_targets = targets
        else:
            print("Data with this dimension cannot be handled")

        self.mat += confusion_matrix(temp_targets, temp_predictions, labels=self.list_classes)

    def get_score(self, predictions, targets, metric='accuracy'):
        if metric == 'accuracy':
            max_index = predictions.max(dim=1)[1]
            correct = (max_index == targets).sum()
            accuracy = int(correct.data) / len(targets)
            return accuracy
        else:
            raise NotImplementedError('Metric not available.')

    def get_notable_indices(self, matrix, top_k=5):
        """
        Return dict of top k and bottom k features useful from matrix.

        Args:
            matrix: 1d numpy array
            top_k: defaults to top and bottom 5 indices
        """
        important_features = matrix.argsort()[-top_k:][::-1]
        unimportant_features = matrix.argsort()[:-1][:top_k]
        return {'important_indices': important_features,
                'unimportant_indices': unimportant_features}

    def round_list(self, raw_list, decimals=4):
        """
        Return the same list with each item rounded off.

        Args:
            raw_list: float list
            decimals: how many decimal points to round to

        Returns: the rounded list
        """
        return [round(item, decimals) for item in raw_list]

    def get_confusion_matrix(self, predictions, targets):
        """
        Calculate the confusion matrix for classification network predictions.

        Args:
            predictions: the class matrix predicted by the network.
                    Does not take one hot vectors.
            targets: the class matrix of the ground truth
                    Does not take one hot vectors.

        Returns: the confusion matrix
        """
        print(type(predictions))
        if len(predictions.shape) == 2:
            predictions = predictions[:, 0]
        if len(targets.shape) == 2:
            targets = targets[:, 0]
        return confusion_matrix(y_true=targets,
                                y_pred=predictions)

    def get_one_hot(self, in_matrix):
        """
        Reformat truth matrix to same size as the output of the dense network.

        Args:
            in_matrix: the categorized 1D matrix

        Returns: a one-hot matrix representing the categorized matrix
        """
        if in_matrix.dtype.name == 'category':
            custum_array = in_matrix.cat.codes

        elif isinstance(in_matrix, np.ndarray):
            custum_array = in_matrix

        else:
            raise ValueError("Input matrix cannot be converted.")

        lb = LabelBinarizer()
        return np.array(lb.fit_transform(custum_array), dtype='float32')

    # TODO: class # should correspond with self.num_class
    def get_class(self, in_matrix):
        """
        Reformat truth matrix to be the classes in a 1D array.

        Args:
            in_matrix: one-hot matrix

        Returns: 2D Class array
        """
        in_matrix= in_matrix.detach()
        if in_matrix.shape[1] > 1:
            return np.expand_dims(np.argmax(in_matrix, axis=1), axis=1)
        elif in_matrix.shape[1] == 1:
            return np.around(in_matrix)

    # TODO: Modify to use val loader
    def run_test(self, model, test_x, test_y, figure_path=None, plot=False):
        """
        Will conduct the test suite to determine model strength.

        Args:
            test_x: data the model has not yet seen to predict
            test_y: corresponding truth vectors
            figure_path: string, folder to place images in.
            plot: bool, determines if graphs should be plotted when ran.
        """
        if model._num_classes is None or model._num_classes == 0:
            raise ValueError('There\'s no classification layer')

        if test_y.shape[1] > 1:
            test_y = self.get_class(test_y)  # Y is in one hot representation

        raw_prediction = model.forward_pass(input_data=test_x,
                                            convert_to_class=False)
        class_prediction = self.get_class(raw_prediction)

        confusion_matrix = self.get_confusion_matrix(
            predictions=class_prediction,
            targets=test_y
        )

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

        print ('{} test\'s results'.format(model.name))

        print ('TP: {}'.format(tp)),
        print ('FP: {}'.format(fp)),
        print ('TN: {}'.format(tn)),
        print ('FN: {}'.format(fn))

        print ('\nAccuracy: {}'.format(accuracy))

        print ('Sensitivity:'),
        print(self.round_list(sens, decimals=3))
        print ('\tMacro Sensitivity: {:.4f}'.format(sens_macro))

        print ('Specificity:'),
        print(self.round_list(spec, decimals=3))
        print ('\tMacro Specificity: {:.4f}'.format(spec_macro))

        print ('DICE:'),
        print(self.round_list(dice, decimals=3))
        print ('\tAvg. DICE: {:.4f}'.format(np.average(dice)))

        print ('Positive Predictive Value:'),
        print(self.round_list(ppv, decimals=3))
        print ('\tMacro Positive Predictive Value: {:.4f}'.format
            (ppv_macro))

        print ('Negative Predictive Value:'),
        print(self.round_list(npv, decimals=3))
        print ('\tMacro Negative Predictive Value: {:.4f}'.format
            (npv_macro))

        print ('F1-score:'),
        print(self.round_list(f1, decimals=3))
        print ('\tMacro f1-score: {:.4f}'.format(f1_macro))
        print('')

        all_class_auc = []
        for i in range(model._num_classes):
            if model._num_classes == 1:      
                fpr, tpr, _ = skl_metrics.roc_curve(test_y,
                                                        raw_prediction.detach(),
                                                        pos_label=1)
            else:
                fpr, tpr, _ = skl_metrics.roc_curve(test_y,
                                                        raw_prediction[:, i].detach(),
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