__author__ = 'Caitrin'
"""
this will define any metrics like average etc, and reutrn

"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn import metrics as scipy_metrics

class Metrics(object):

    def average(self):
        pass

    def get_accuracy(preds, targets):
        preds = preds.flatten()
        targets = targets.flatten()
        correct = np.sum(preds == targets)
        return correct / len(targets)

    def get_cross_entropy_loss(preds, targets):
        return F.binary_cross_entropy(
                  Variable(torch.from_numpy(preds)),
                  Variable(torch.from_numpy(targets).float())).data[0]

    def get_precision(preds, targets):
        return scipy_metrics.precision_score(targets.flatten(), preds.flatten())

    def get_roc_score(preds, targets):
        return scipy_metrics.roc_auc_score(targets.flatten(), preds.flatten())
