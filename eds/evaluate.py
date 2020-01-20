from sklearn.model_selection import cross_validate
from eds import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from typing import Callable
from eds import _default_models
import numpy as np


def get_cross_val_ci(model, x: np.array, y: np.array, loss=True,
                     metric: Callable[[np.array, np.array], float] = None,
                     folds=5) -> dict:
    """
    Returns performance evaluation of a model based on default or custom metric.
    Constructs a 95% confidence interval for an estimation of performance.

    Parameters
    ----------
    model
        Model to evaluate cross-validation performance on.

    x : np.array
        Set of feature vectors.

    y : np.array
        Respective outputs of feature vectors.

    loss: bool, default True
        Whether this is a loss function (should be minimized) or opposite.

    metric : Callable[[np.array, np.array], float]
        Custom metric to use for evaluating cross-validation performance.

    folds: int
        Number of cross-validation folds.

    Returns
    -------
    np.array
        Dictionary containing performance information.
    """
    scorer_func = make_scorer(metric, greater_is_better=~loss)
    scores = cross_val_score(model, x, y, scoring=scorer_func, cv=folds)
    if not loss:
        scores = scores * -1

    return {'Avg_Score': scores.mean(),
            '95_CI_Low': scores.mean() - scores.std() * 2,
            '95_CI_High': scores.mean() + scores.std() * 2
            }


def evaluate_regressor(y_actual: np.array, y_pred: np.array) -> dict:
    """
    Evaluates a regression model's set of predictions.
    Prints out several performance scores.
    Returns dictionary containing performance information.

    Parameters
    ----------
    y_actual : np.array
        True labels of feature vectors.

    y_pred : np.array
        Predicted labels of feature vectors.

    Returns
    -------
    dict
        Performance information for each main metric.
    """
    r2 = 1 - metrics.r2_error(y_actual, y_pred)
    mse = metrics.mse(y_actual, y_pred)
    rmse = mse ** (1 / 2)

    return {
        'R2': round(r2, 4),
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4)
    }


def try_many_regressors(x: np.array, y: np.array, loss=True, folds=5,
                        metric: Callable[[np.array, np.array], float] = metrics.mse) -> dict:
    """
    Tries a set of different regression models and returns
    report on the model's performances.

    Parameters
    ----------
    x : np.array
        Set of feature vectors.

    y : np.array
        Respective outputs of feature vectors.

    loss : bool, optional, by default True
        Whether the metric is a loss function. (Smaller means better)

    folds : int, optional, by default 5
        Number of folds for cross-validation

    metric : Callable[[np.array, np.array], float], optional, by default metrics.mse
        Custom metric to minimize, by default mean squared error.

    Returns
    -------
    dict
        Dictionary containing result information including a 
        confidence interval of the model's score as well as the
        unfitted model.
    """
    def first(s):
        return s[0] if len(s) > 1 else s

    regressors = _default_models.get_default_regressors()

    scores = [{} for _ in range(len(regressors))]
    scorer_func = make_scorer(metric, greater_is_better=~loss)

    for i, model in enumerate(regressors):
        print('Running k-fold cross validation for', model.__class__.__name__)
        scores[i]['model_name'] = model.__class__.__name__
        scores[i]['model'] = model
        current_scores = cross_validate(model, x, y, scoring=scorer_func, cv=folds)['test_score']
        scores[i]['mean_score'] = current_scores.mean()
        scores[i]['score_95_CI_Low'] = current_scores.mean() - (current_scores.std() * 2)
        scores[i]['score_95_CI_High'] = current_scores.mean() + (current_scores.std() * 2)

    return scores


def get_best_regressor(x: np.array, y: np.array, loss=True, folds=5,
                       metric: Callable[[np.array, np.array], float] = metrics.mse):
    """
    Tries a suite of different regressors and returns the best performing one.

    Parameters
    ----------
    x : np.array
        Set of feature vectors.

    y : np.array
        Respective outputs of feature vectors.

    loss : bool, optional, by default True
        Whether the metric is a loss function. (Smaller means better)

    folds : int, optional, by default 5
        Number of folds for cross-validation

    metric : Callable[[np.array, np.array], float], optional, by default metrics.mse
        Custom metric to minimize, by default mean squared error.

    Returns
    -------
    model
        Sklearn like model that performed best compared to a wide selection of models.
    """
    results = try_many_regressors(x, y, loss, folds, metric)
    scores = [res['mean_score'] for res in results]
    return results[np.argmin(scores)]['model']
