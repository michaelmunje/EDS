from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from scipy.optimize import differential_evolution
from eds import metrics
from eds import _default_models
from typing import Callable
import numpy as np
from numpy.linalg import norm
from abc import ABC, abstractmethod


class Ensemble(ABC):
    """
    Base class for an sklearn model-like wrapper for an ensemble of models.
    """
    @abstractmethod
    def __init__(self, models: [] = None, names: [str] = None, weights: np.array = None):
        self.models = models if models else self.get_default_models()
        self.weights = weights if weights else [1/len(self.models)] * len(self.models)
        self.names = names if models else Ensemble.get_default_models_names()

    @abstractmethod
    def predict(self, x: np.array) -> np.array:
        pass

    @abstractmethod
    def _get_model_pred(self, model_index: int, X: np.array):
        pass

    @staticmethod
    @abstractmethod
    def get_default_models() -> []:
        pass

    @staticmethod
    @abstractmethod
    def get_default_models_names() -> [str]:
        pass

    def fit(self, X: np.array, Y: np.array) -> None:
        """
        Fits each of the models in the ensemble to the data.
        This is usually the last step and assumes the weights
        have been optimized already.

        Parameters
        ----------
        X : np.array
            Set of feature vectors.

        Y : np.array
            Set of respective outputs of the feature vectors.
        """
        for model in self.models:
            model.fit(X, Y)

    def optimal_fit_cv(self, X: np.array, Y: np.array) -> None:
        """
        Optimizes the weights of the vector using cross-validation,
        then fits to all of the data.

        Parameters
        ----------
        X : np.array
            Set of feature vectors.

        Y : np.array
            Set of respective outputs of the feature vectors.
        """
        self.optimize_weights_cv(X, Y)
        self.fit(X, Y)

    def __normalize_weights(self):
        """
        This normalizes the weight vector.
        Specifically used during weight optimization.
        """
        result = norm(self.weights, 1)
        self.weights = self.weights / result if not result == 0.0 else self.weights

    def optimize_weights_cv(self, X: np.array, Y: np.array, iterations: int = 1000) -> None:
        """
        Optimizes the weights of the ensemble by using cross-validation.

        Parameters
        ----------
        X : np.array
            Set of feature vectors.

        Y : np.array
            Set of respective outputs of the feature vectors.

        iterations : int, optional
            Number of optimization steps, by default 1000
        """
        print('Getting cross-validation results...')
        model_preds, y_holdout_true = self.__get_cv_holdout_results(X, Y)

        print('Optimizing weights using results...')

        bounds = [(0.0, 1.0)] * len(self.models)
        result = differential_evolution(self.__evaluate_ensemble_weights, bounds,
                                        args=(model_preds, y_holdout_true),
                                        maxiter=iterations, tol=1e-7)
        self.weights = result['x']
        self.__normalize_weights()

    def get_model_importances(self):
        """
        Returns a list of important models.
        Importance calculated using the weights.
        If names were not provided, simply returns the weights.

        Returns
        -------
        []
            Returns a list of important models.
        """
        if self.names:
            tuples = list(zip(self.names, [round(w, 3) for w in self.weights]))
            return sorted(tuples, key=lambda x: x[1], reverse=True)
        return self.weights

    def get_cv_results(self, X: np.array, Y: np.array, num_splits: int = 10) -> (np.array, np.array):
        model_preds, y_holdout_true = self.__get_cv_holdout_results(X, Y)
        return self.__evaluate_ensemble_weights(self.weights, model_preds, y_holdout_true)

    def __evaluate_ensemble_weights(self, weights: np.array, model_preds: np.array, y_test: np.array) -> float:
        """
        Private method used by optimization process to get performance of the current weight vector.

        Parameters
        ----------
        weights : np.array
            New weights to evaluate on.

        model_preds : np.array
            Array of model's predictions for X

        y_test : np.array
            True outputs of X

        Returns
        -------
        float
            Performance on metric.
        """
        self.weights = weights
        self.__normalize_weights()
        y_pred = sum(x * y for x, y in zip(model_preds, self.weights))
        return self.metric(y_test, y_pred)

    def __get_cv_holdout_results(self, X: np.array, Y: np.array, num_splits: int = 10) -> (np.array, np.array):
        """
        Uses cross-validation to get prediction of each holdout set.

        Parameters
        ----------
        X : np.array
            Set of feature vectors.

        Y: np.array
            Respective outputs of feature vectors.

        num_splits: int
            Number of k splits, by default 10

        Returns
        -------
        (np.array, np.array)
            Returns the set of holdout predictions of each model, and the true outputs
        """
        if len(np.unique(Y)) <= 2:
            cv = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=1337)
        else:
            cv = KFold(n_splits=num_splits, shuffle=True, random_state=1337)
        results = [[] for _ in range(len(self.models))]
        Y_holdout_true = []
        for train, test in cv.split(X, Y):
            Y_holdout_true.extend(Y[test])
            for j in range(len(self.models)):
                self.models[j].fit(X[train], Y[train])
                results[j].extend(self._get_model_pred(j, X[test]))
        return np.array(results), np.array(Y_holdout_true)


class EnsembleClassifier(Ensemble):

    def __init__(self, models=None, names: [str] = None, weights: np.array = None,
                 metric: Callable[[np.array, np.array], float] = metrics.roc_auc_error):
        """
        Class for ensemble of classifiers which is an sklearn-like model but for an ensemble of classifiers.

        Parameters
        ----------
        models : [], optional
            List (or sequence-like object) containing predefined sklearn-like classifiers
            If no models provided, uses a default list of models.

        names : [type], optional
            List of strings containing the names of the models. Defaults to None.
            If no models provided, uses a default list of model names.

        weights : np.array, optional
            Predefined set of weights.
            Defaults to equally weighted ensemble if no argument provided.

        metric : Callable[[np.array, np.array], float], optional
            Function for computing a metric to minimize.
            A wrapper function can multiply a function to maximize by -1 to minimize instead.
            Defaults to minimize the error of ROC Area under the curve
        """
        self.metric = metric
        super().__init__(models, names, weights)

    def predict(self, x: np.array) -> np.array:
        """
        Predicts classes of feature vectors with the ensemble.
        Note: Binary output is only supported at this time.

        Parameters
        ----------
        x : np.array
            Set of feature vectors to predict classes for.

        Returns
        -------
        np.array
            Ensemble's prediction of binary classes.
        """
        model_preds = [model.predict(x) for model in self.models]
        preds = sum(x * y for x, y in zip(model_preds, self.weights))
        return np.array([1 if pred >= 0.5 else 0 for pred in preds])

    def predict_proba(self, x: np.array) -> np.array:
        """
        Predicts the class probability vector for the feature vectors with the ensemble.

        Parameters
        ----------
        x : np.array
            Set of feature vectors.

        Returns
        -------
        np.array
            Prediction for the classes probabilities for each feature vector.
        """
        model_preds = [model.predict_proba(x) for model in self.models]
        preds = sum(x * y for x, y in zip(model_preds, self.weights))
        return np.array(preds)

    def _get_model_pred(self, model_index: int, x: np.array) -> np.array:
        """
        Gets the prediction of a fecture vector of a specific model in the ensemble.

        Parameters
        ----------
        model_index : int
            Index in ensemble to get classifier prediction for.

        X : np.array
            Input array consisting of feature vector to predict output for.
        """
        return self.models[model_index].predict_proba(x)[:, 1]

    @staticmethod
    def get_default_models() -> []:
        """
        Gets an array consisting of predefined classifiers.
        The specific classifiers can be found in :ref:`User Guide <ensemble>`

        Returns
        -------
        []
            An array consisting of predefined classifiers.
        """
        return _default_models.get_default_classfiers()

    @staticmethod
    def get_default_models_names() -> [str]:
        """
        Gets an array consisting of predefined classifiers' names.
        The specific classifiers can be found in :ref:`User Guide <ensemble>`

        Returns
        -------
        []
            An array consisting of string names of predefined classifiers.
        """
        return _default_models.get_default_classfiers_names()


class EnsembleRegressor(Ensemble):

    def __init__(self, models=None, names: [str] = None, weights: np.array = None,
                 metric: Callable[[np.array, np.array], float] = metrics.mse):
        """
        Class for ensemble of regressors which is an sklearn-like model but for an ensemble of regressors.

        Parameters
        ----------
        models : [], optional
            List (or sequence-like object) containing predefined sklearn-like regressors
            If no models provided, uses a default list of models.

        names : [type], optional
            List of strings containing the names of the models. Defaults to None.
            If no models provided, uses a default list of model names.

        weights : np.array, optional
            Predefined set of weights.
            Defaults to equally weighted ensemble if no argument provided.

        """
        self.metric = metric
        super().__init__(models, names, weights)

    def predict(self, x: np.array) -> np.array:
        """
        Predicts output scalar of feature vectors with the ensemble.

        Parameters
        ----------
        x : np.array
            Set of feature vectors to predict output scalar for.

        Returns
        -------
        np.array
            Ensemble's regression prediction for each feature vector.
        """
        model_preds = [model.predict(x) for model in self.models]
        preds = sum(x * y for x, y in zip(model_preds, self.weights))
        return np.array(preds)

    def _get_model_pred(self, model_index: int, X: np.array) -> np.array:
        """
        Gets the prediction array for a specific model in the ensemble.

        Parameters
        ----------
        model_index : int
            Index in ensemble to get model prediction for.

        X : np.array
            Input array consisting of feature vector to predict output for.
        """
        return self.models[model_index].predict(X)

    @staticmethod
    def get_default_models() -> []:
        """
        Gets an array consisting of predefined regressors.
        The specific regressors can be found in :ref:`User Guide <ensemble>`

        Returns
        -------
        []
            An array consisting of predefined regressors.
        """
        return _default_models.get_default_regressors()

    @staticmethod
    def get_default_models_names() -> [str]:
        """
        Gets an array consisting of predefined regressors' names.
        The specific regressors can be found in :ref:`User Guide <ensemble>`

        Returns
        -------
        []
            An array consisting of string names of predefined regressors.
        """
        return _default_models.get_default_regressors_names()
