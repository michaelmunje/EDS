from sklearn.model_selection import StratifiedKFold
from scipy.optimize import differential_evolution
import metrics
import _default_models
from typing import Callable
import numpy as np
from numpy.linalg import norm
from abc import ABC, abstractmethod


class Ensemble(ABC):

    @abstractmethod
    def __init__(self, models, names: [str] = None, weights: np.array = None):
        self.models = models
        self.weights = [1/len(self.models)] * len(self.models) if not weights else weights
        self.names = names

    @abstractmethod
    def predict(self, x: np.array) -> np.array:
        pass

    @abstractmethod
    def _get_model_pred(self, model_index: int, X: np.array):
        pass

    @staticmethod
    @abstractmethod
    def get_default_models():
        pass

    @staticmethod
    @abstractmethod
    def get_default_models_names():
        pass

    def fit(self, X: np.array, Y: np.array) -> None:
        for model in self.models:
            model.fit(X, Y)

    def __normalize_weights(self):
        result = norm(self.weights, 1)
        self.weights = self.weights / result if not result == 0.0 else self.weights

    def optimize_weights(self, X: np.array, Y: np.array, iterations: int = 1000) -> None:
        model_preds, y_holdout_true = self.__get_cv_holdout_results(X, Y)

        bounds = [(0.0, 1.0)] * len(self.models)
        result = differential_evolution(self.__evaluate_ensemble_weights, bounds,
                                        args=(model_preds, y_holdout_true),
                                        maxiter=iterations, tol=1e-7)
        self.weights = result['x']
        self.__normalize_weights()

    def get_model_importances(self):
        if self.names:
            tuples = list(zip(self.names, [round(w, 3) for w in self.weights]))
            return sorted(tuples, key=lambda x: x[1], reverse=True)
        return sorted(self.weights, reverse=True)

    def __evaluate_ensemble_weights(self, weights: np.array, model_preds: np.array, y_test: np.array) -> float:
        self.__normalize_weights()
        y_pred = sum(x * y for x, y in zip(model_preds, weights))
        return self.metric(y_test, y_pred)

    def __get_cv_holdout_results(self, X: np.array, Y: np.array, prob=True, num_splits=10) -> (np.array, np.array):
        cv = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=1337)
        results = [ [] for _ in range(len(self.models))]
        Y_holdout_true = []
        for train, test in cv.split(X, Y):
            Y_holdout_true.extend(Y[test])
            for j in range(len(self.models)):
                self.models[j].fit(X[train], Y[train])
                results[j].extend(self._get_model_pred(j, X[test]))
        return np.array(results), np.array(Y_holdout_true)


class EnsembleClassifier(Ensemble):

    def __init__(self, models, names: [str] = None, weights: np.array = None, 
                 metric: Callable[[np.array, np.array], float] = metrics.roc_auc_error):

        self.metric = metric
        super().__init__(models, names, weights)

    def predict(self, x: np.array, prob=False) -> np.array:
        model_preds = [model.predict(x) if not prob else model.predict_proba(x) for model in self.models]
        preds = sum(x * y for x, y in zip(model_preds, self.weights))
        return np.array(preds) if prob else np.array([1 if pred >= 0.5 else 0 for pred in preds])

    def predict_proba(self, x: np.array) -> np.array:
        return self.predict(x, prob=True)

    def _get_model_pred(self, model_index: int, x: np.array) -> np.array:
        return self.models[model_index].predict_proba(x)[:, 1]

    @staticmethod
    def get_default_models() -> []:
        return _default_models.get_default_classfiers()

    @staticmethod
    def get_default_models_names() -> [str]:
        return _default_models.get_default_classfiers_names()


class EnsembleRegressor(Ensemble):

    def __init__(self, models, names: [str] = None, weights: np.array = None,
                 metric: Callable[[np.array, np.array], float] = metrics.mse):

        self.metric = metric
        super().__init__(models, names, weights)

    def predict(self, x: np.array) -> np.array:
        model_preds = [model.predict(x) for model in self.models]
        preds = sum(x * y for x, y in zip(model_preds, self.weights))
        return np.array([1 if pred >= 0.5 else 0 for pred in preds])

    def _get_model_pred(self, model_index: int, X: np.array) -> np.array:
        return self.models[model_index].predict(X)

    @staticmethod
    def get_default_models() -> []:
        return _default_models.get_default_regressors()

    @staticmethod
    def get_default_models_names() -> [str]:
        return _default_models.get_default_regressors_names()
