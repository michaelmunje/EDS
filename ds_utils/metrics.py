import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

def false_positive_rate(Y: np.array, Y_hat: np.array) -> np.array:
    return 1 - recall_score(Y,Y_hat)

def false_negative_rate(Y: np.array, Y_hat: np.array) -> np.array:
    return 1 - recall_score(1-Y, 1-Y_hat)

def avg_error(Y: np.array, Y_hat: np.array) -> np.array:
    Y_hat = np.array([1 if pred >= 0.5 else 0 for pred in Y_hat])
    return (false_positive_rate(Y_hat, Y) + false_negative_rate(Y_hat, Y)) / 2

def roc_auc_error(Y, Y_hat):
    fpr, tpr, thresholds = roc_curve(Y, Y_hat)
    return 1 - auc(fpr, tpr)