import torch
from sklearn.metrics import roc_auc_score

from .metrics import *


def eval_metrics(target, pred_prob, threshold=0.5):
    """
    Calculate a group of evaluation metrics for a model's prediction for target binary labels.

    :param target: must be numpy.ndarray or torch.Tensor.
    :param pred_prob: should be the probabilities, instead of binary classification results.

    :return: dict
    """
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    if isinstance(pred_prob, torch.Tensor):
        pred_prob = pred_prob.cpu().numpy()

    pred = (pred_prob >= threshold).astype(int)

    acc = accuracy(pred, target)
    fpr = false_positive_rate(pred, target)
    fnr = false_negative_rate(pred, target)
    rec = recall(pred, target)
    prc = precision(pred, target)
    f1 = f1_score(pred, target)
    auroc = roc_auc_score(target, pred_prob)
    result_dict = {'acc': acc, 'fpr': fpr, 'fnr': fnr, 'rec': rec, 'prc': prc, 'f1': f1, 'auroc': auroc}

    return result_dict
