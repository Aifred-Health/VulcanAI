import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn import metrics as scipy_metrics

class Metrics(object):

    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((num_class, num_class))

    def _fast_hist(self, prediction, target, num_class):
        mask = (target >= 0) & (target < num_class)
        hist = np.bincount(
                            num_class * target[mask].astype(int) + prediction[mask],
                            minlength=num_class ** 2,
                            ).reshape(num_class, num_class)
        return hist
    def update(self, predictions, targets):
        for prediction, target in zip(predictions, targets):
            self.confusion_matrix += self._fast_hist(
                prediction.flatten(), target.flatten(), self.num_class)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        cls_iu = dict(zip(range(self.num_class), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
            )  

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def average(self):
        pass

    def get_accuracy(self, predictions, targets):
        max_index = predictions.max(dim=1)[1]
        correct = (max_index == targets).sum()
        accuracy = int(correct.data) / len(targets)
        return correct, accuracy

    def get_precision(predictions, targets):
        return scipy_metrics.precision_score(targets.flatten(), predictions.flatten())

    def get_roc_score(predictions, targets):
        return scipy_metrics.roc_auc_score(targets.flatten(), predictions.flatten())
