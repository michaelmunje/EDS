from sklearn.model_selection import cross_validate
from eds import metrics
from sklearn.model_selection import cross_val_score
from typing import Callable
from eds import _default_models
import numpy as np


def evaluate_model_cross_val(model, x: np.array, y: np.array,
                             metric: Callable[[np.array, np.array], float]) -> np.array:
    """
    Prints out performance evaluation of a model based on default or custom metric.

    Parameters
    ----------
    model
        Model to evaluate cross-validation performance on.

    x : np.array
        Set of feature vectors.

    y : np.array
        Respective outputs of feature vectors.

    metric : Callable[[np.array, np.array], float]
        Custom metric to use for evaluating cross-validation performance.

    Returns
    -------
    np.array
        All scores on holdout sets during cross-validation.
    """
    scores = cross_val_score(model, x, y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 95% confidence interval
    return scores


def evaluate_regressor(y_actual: np.array, y_pred: np.array, metric_func: Callable[[np.array, np.array], float]) -> [float]:
    """
    Evaluates a regression model's set of predictions.
    Prints out several performance scores.
    Returns list containing all performances.

    Parameters
    ----------
    y_actual : np.array
        True labels of feature vectors.

    y_pred : np.array
        Predicted labels of feature vectors.

    metric_func : Callable[[np.array, np.array], float]
        Custom performance metric to use, by default None.

    Returns
    -------
    [float]
        List of performance scores.
    """
    r2 = metrics.r2(y_actual, y_pred)
    mse = metrics.mse(y_actual, y_pred)
    rmse = mse ** (1 / 2)
    all_metrics = [r2, mse, rmse]

    print('R2\t: ', round(r2, 4))
    print('MSE\t: ', round(mse, 4))
    print('RMSE\t: ', round(rmse, 4))

    if metric_func:
        custom_metric = metric_func(y_actual, y_pred)
        all_metrics.append(custom_metric)
        print('CUSTOM METRIC\t: ', round(custom_metric, 4))

    return all_metrics


def try_many_regressors(x: np.array, y: np.array, metric: Callable[[np.array, np.array], float] = metrics.mse,
                        metric_max_better: bool = True) -> None:
    """
    Tries a set of different regression models and reports on the best performing model.

    Parameters
    ----------
    x : np.array
        Set of feature vectors.

    y : np.array
        Respective outputs of feature vectors.

    metric : Callable[[np.array, np.array], float]
        Custom metric to minimize, by default mean squared error.
        A wrapper function can multiply a function to maximize by -1 to minimize instead.

    Returns
    -------
        Returns the best performing fitted model.
    """
    def first(s):
        return s[0] if len(s) > 1 else s
    regressors = _default_models.get_default_regressors()

    scores = np.zeros(len(regressors))

    for i, r in enumerate(regressors):
        print('Running k-fold cross validation for', r.__class__.__name__)
        scores[i] = cross_validate(r, x, y, metric)

    print('Best performing model: ', regressors[first(np.argmin(scores))].__class__.__name__)
    print('Best', metric.__name__, ':', np.amin(scores))
    return regressors[first(np.argmin(scores))]
