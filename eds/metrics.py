import numpy as np
from sklearn.metrics import precision_score
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

    Calculates the false positive rate of a set of predictions.
    >>> y_true = [1, 0]
    >>> y_pred = [1, 1]
    >>> false_positive_rate(np.array(y_true), np.array(y_pred))
    0.5
    """
    return 1 - precision_score(Y, Y_hat)


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

    Calculates the false negative rate of a set of predictions.
    >>> y_true = [1, 0, 1, 1]
    >>> y_pred = [0, 0, 0, 0]
    >>> false_negative_rate(np.array(y_true), np.array(y_pred))
    0.75
    """
    return 1 - precision_score(1-Y, 1-Y_hat)


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

    Returns the average of the FPR and the FNR.
    >>> y_true = [1, 0, 0]
    >>> y_pred = [1, 1, 0]
    >>> avg_error(np.array(y_true), np.array(y_pred))
    0.25
    """
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

    Returns the ROC AUC Error, (1 - AUC).
    >>> y_true = [1, 1, 0, 1, 1]
    >>> y_pred = [0.8, 0.1, 0.6, 0.7, 0.7]
    >>> roc_auc_error(np.array(y_true), np.array(y_pred))
    0.25
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

    The mean squared error is the average squared distance from predictions and true labels.
    >>> y_true = [5, 15, 10, 5, 15]
    >>> y_pred = [10, 20, 5, 0, 10]
    >>> mean_squared_error(np.array(y_true), np.array(y_pred))
    25.0
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

    Returns the error of the R2 (1 - R2)
    >>> y_true = [9, 19, 9]
    >>> y_pred = [10, 20, 10]
    >>> round(r2_error(np.array(y_true), np.array(y_pred)), 2)
    0.05
    """
    return 1 - r2_score(Y, Y_hat)
