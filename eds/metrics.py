import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def false_positive_rate(Y: np.array, Y_hat: np.array) -> float:
    """
    False positive rate of the predictions.
    Also known as type I error rate.

    Parameters
    ----------
    Y : np.array
        Array of true labels

    Y_hat : np.array
        Array of predicted labels

    Returns
    -------
    float
        False positive rate
    """
    return 1 - recall_score(Y, Y_hat)


def false_negative_rate(Y: np.array, Y_hat: np.array) -> float:
    """
    False negative rate of the predictions.
    Also known as type II error rate.

    Parameters
    ----------
    Y : np.array
        Array of true labels

    Y_hat : np.array
        Array of predicted labels

    Returns
    -------
    float
        False negative rate
    """
    return 1 - recall_score(1-Y, 1-Y_hat)


def avg_error(Y: np.array, Y_hat: np.array) -> float:
    """
    Average of the FPR and FNR.

    Parameters
    ----------
    Y : np.array
        Array of true labels

    Y_hat : np.array
        Array of predicted labels

    Returns
    -------
    float
        Average of the FPR and FNR.
    """
    Y_hat = np.array([1 if pred >= 0.5 else 0 for pred in Y_hat])
    return (false_positive_rate(Y, Y_hat) + false_negative_rate(Y, Y_hat)) / 2


def roc_auc_error(Y: np.array, Y_hat: np.array) -> float:
    """
    Returns the error rate of the ROC AUC.
    In other words, 1 - ROC AUC

    Parameters
    ----------
    Y : np.array
        Array of true labels

    Y_hat : np.array
        Array of predicted labels
        Important note: This should be a probability vector
        of probability that X's label is the positive class.

    Returns
    -------
    float
        Returns the error rate of the ROC AUC.
    """
    fpr, tpr, thresholds = roc_curve(Y, Y_hat)
    return 1 - auc(fpr, tpr)


def mse(Y: np.array, Y_hat: np.array) -> float:
    """
    Mean squared error of the prediction.
    Equivalent to the residual sum of squares normalized by size.

    Parameters
    ----------
    Y : np.array
        Array of true labels

    Y_hat : np.array
        Array of predicted labels

    Returns
    -------
    float
        Mean squared error of the prediction.
    """
    return mean_squared_error(Y, Y_hat)


def r2_error(Y: np.array, Y_hat: np.array) -> float:
    """
    R2 score error of the prediction.
    See `wiki <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_
    for more info.
    Since it returns error, returns 1 - R2

    Parameters
    ----------
    Y : np.array
        Array of true labels

    Y_hat : np.array
        Array of predicted labels

    Returns
    -------
    float
        Error rate of the R2 score.
    """
    return 1 - r2_score(Y, Y_hat)
