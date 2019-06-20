"""
Author: Michael Munje
Github: https://github.com/michaelmunje/ds_utils
"""

from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from typing import Callable
import seaborn as sns
import copy
import pandas as pd
import numpy as np


def apply_scale(x_train: np.array, x_test: np.array, scale_type: str = 'Standard') -> (np.array, np.array):
    """
    Scales the data according to the distribution of x_train
    :param x_train: numpy.array to scale via its distribution
    :param x_test: numpy.array to scale according to x_test's distribution
    :param scale_type: Which scaling type to use. Options: 'Standard', 'Robust', 'MinMax'. Default = 'Standard'
    :return: x_train and x_test properly scaled.
    """

    if scale_type == 'Standard':
        scaler = StandardScaler()
    elif scale_type == 'Robust':
        scaler = RobustScaler()
    elif scale_type == 'MinMax':
        scaler = RobustScaler()
    else:
        print("Invalid input. Defaulting to StandardScaler")
        scaler = StandardScaler()

    scaler.fit(x_train)
    return scaler.transform(x_train), scaler.transform(x_test)


def apply_pca(x_train: np.array, x_test: np.array, n_comps: float) -> (np.array, np.array):
    """
    Apply PCA to the data according to the distribution of x_train
    :param x_train: numpy.array to scale via its distribution
    :param x_test: numpy.array to scale according to x_test's distribution
    :param n_comps: Either the number of principal components, or the variance to be kept in the components.
    :return: x_train and x_test with PCA applied.
    """

    pca = PCA(n_components=n_comps)
    pca.fit(x_train)
    return pca.transform(x_train), pca.transform(x_test)


def plot_hist_distribution(df: pd.Series, col: str) -> None:
    """
    Plots a histogram of the column's distribution
    :param df: pandas DataFrame to extract column from
    :param col: string that represents column
    """

    df[col].hist(bins=50, facecolor='red', alpha=0.5)
    plt.title('Distribution of ' + col)
    plt.xlabel(col)
    plt.ylabel('Freq.')
    fig = plt.gcf()
    fig.set_size_inches(5, 5)
    plt.show()


def plot_relationship(df: pd.DataFrame, feature1: str, feature2: str) -> None:
    """
    Plots the relationship of 2 features to each other
    :param df: pandas DataFrame where features are contained
    :param feature1: feature 1's column name
    :param feature2: feature 2's column name
    """

    plt.scatter(df[feature1], df[feature2],
                s=50, color='blue', label='Normal')
    plt.grid()
    plt.title('Pairwise Feature Relationship', fontsize=15)
    plt.xlabel(feature1, fontsize=20)
    plt.ylabel(feature2, fontsize=20)
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    plt.show()


def plot_binary_feature_distribution(df, feature, class_name, is_hist=False):
    """
    Plots 2 distributions of a feature: class_name or not class_name
    This helps us to see where the difference in distributions
    :param df: pandas DataFrame to
    :param feature: Which feature to check the distributions of
    :param class_name: The class we draw an independent distribution of
    :param is_hist: Plot a histogram or KDE. Default = False
    """
    bins = np.linspace(df[feature].min(), df[feature].max(), 30)
    if is_hist:
        sns.distplot([df[df[class_name] == 0][feature], df[df[class_name] == 1][feature]], hist=True,
                     norm_hist=True, kde=False,
                     bins=bins, color=['blue', 'red'], label=['Not ' + class_name, class_name],
                     hist_kws={'edgecolor': 'black'})
    else:
        sns.kdeplot(df[df['Survived'] == 0][feature], shade=True, color='Blue', label='Not ' + class_name)
        sns.kdeplot(df[df['Survived'] == 1][feature], shade=True, color='Red', label=class_name)
    plt.legend(prop={'size': 13}, title=class_name)
    plt.title('Density Plot of ' + feature, fontsize=15)
    plt.xlabel(feature, fontsize=20)
    plt.ylabel('Density', fontsize=20)
    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    plt.show()


def get_nan_col_proportions(df: pd.DataFrame, lowest_proportion: float = 0.0) -> [(str, float)]:
    """
    Prints out all columns with NaN values that exceed a specific proportion (default 0.0)
    :param df: pandas DataFrame to look into.
    :param lowest_proportion: float that is the lowest proportions that we print.
    :return: None
    """

    values = list(zip(list(df.isnull().columns), list(df.isnull().any())))
    filtered = list(filter(lambda x: x[1][1] == True, enumerate(values)))
    contains_nan = [y for x, y in filtered]
    proportion_nan = [round(sum(df[x].isnull()) / len(df[x]), 3) for x, y in contains_nan]
    proportion_nan = [(x[0], proportion_nan[i]) for i, x in enumerate(contains_nan)]
    nan_prop_list = list()
    for col, propo_nan in proportion_nan:
        if abs(propo_nan) > lowest_proportion:
            nan_prop_list.append((col, propo_nan))
    return nan_prop_list


def remove_nan_cols(df: pd.DataFrame, prop_threshold: float = 0.0) -> pd.DataFrame:
    """
    Prints out all columns with NaN values that exceed a specific proportion (default 0.0)
    :param df: pandas DataFrame to look into.
    :param prop_threshold: float that is the lowest proportion that we delete
    :return: pandas DataFrame with removed nans
    """

    nan_props = get_nan_col_proportions(df, prop_threshold)
    names = [name for name, _ in nan_props]
    df.drop(columns=names, inplace=True)
    return df


def print_moderate_correlations(df: pd.DataFrame, col_to_correlate: str, moderate_value: float = 0.4) -> None:
    """
    Prints out all correlations deemed as moderate (0.4, or set by parameter).
    :param df: pandas DataFrame to look into.
    :param col_to_correlate: String that represents column we want to check correlations with.
    :param moderate_value: Which correlation value we deem as moderate (default 0.4).
    """

    if df[col_to_correlate].dtype.name == 'category':
        df[col_to_correlate] = df[col_to_correlate].cat.codes
    corrs = df[df.columns].corr()
    cols = corrs.columns
    corrs_value = corrs[col_to_correlate]
    for col, corr_value in zip(cols, corrs_value):
        if abs(corr_value) > moderate_value and col != col_to_correlate:
            print(col, ': ', corr_value)


def remove_weak_correlations(df1: pd.DataFrame, df2, y, weak_threshold: float = 0.1) -> (pd.DataFrame, pd.DataFrame):
    """
    Removes weak correlations
    :param df: pandas DataFrame to remove columns from.
    :param col_to_correlate: String column name to check correlation with
    :param weak_threshold: float number that counts as an absolute weak threshold
    :return: pandas DataFrame without the columns weakly correlated to target
    """
    col_to_correlate = 'PREDICTOR_TO_CHECK_WEAK_CORRS'
    df1[col_to_correlate] = y
    cols = df1[df1.columns].corr().columns
    corrs = df1[df1.columns].corr()[col_to_correlate]
    strongly_correlated = list()
    for col, corr in zip(cols, corrs):
        if abs(corr) >= weak_threshold and col != col_to_correlate:
            strongly_correlated.append(col)
    df1 = df1.drop(columns=[col_to_correlate])
    for col in df1.columns:
        if col not in strongly_correlated:
            df1.drop(columns=[col], inplace=True)
            df2.drop(columns=[col], inplace=True)
    return df1, df2


def evaluate_classifier(clf, x, y):
    scores = cross_val_score(clf, x, y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


def get_rid_feature(df, df2, feat):
    for column in df:
        if column.startswith(feat):
            df.drop(columns=[column], inplace=True)
            df2.drop(columns=[column], inplace=True)
    return df, df2


def convert_categorical_to_numbers(to_change_df: pd.DataFrame) -> pd.DataFrame:
    """
    Dummifies all category data including objects.
    Assumes the data has been cleaned and the dtypes are consistent.
    :param to_change_df: pandas DataFrame to convert to all numerical
    :return: Dummified input pandas DataFrame
    """
    return pd.get_dummies(convert_objects_to_categories(to_change_df))


def convert_objects_to_categories(to_change_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts all object dtypes into categories
    :param to_change_df: pandas DataFrame to convert objects to categories
    :return: pandas DataFrame with categories instead of objects
    """

    for col, dtype in zip(to_change_df.columns, to_change_df.dtypes):
        if dtype == object:
            to_change_df[col] = to_change_df[col].astype('category')
    return to_change_df


def replace_missing_with_ml(df: pd.DataFrame, col_to_predict: str) -> None:
    """
    Replace the missing values in the given column using machine learning predictions
    :param df: pandas DataFrame to use as features (and predictor column)
    :param col_to_predict: string that represents the predictor column
    :return: pandas DataFrame with filled predictor column values via machine learning
    """
    y = df[col_to_predict].values

    dummified_df = df.copy()

    cols_to_drop = filter(lambda t: t[1], zip(df.columns, df.isnull().any()))
    cols_to_drop = list(x[0] for x in cols_to_drop)
    dummified_df = dummified_df.drop(columns=cols_to_drop)
    dummified_df = convert_categorical_to_numbers(dummified_df)

    dummified_df[col_to_predict] = y

    df_to_model = dummified_df.dropna(subset=[col_to_predict])

    df_to_predict = dummified_df[dummified_df[col_to_predict].isnull()]
    df_to_predict = df_to_predict.drop(columns=[col_to_predict])

    y = df_to_model[col_to_predict]
    x = df_to_model.drop(columns=[col_to_predict]).values

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

    is_classify = True

    if df[col_to_predict].dtype.name == 'object':
        rf = GradientBoostingClassifier(n_estimators=3000, learning_rate=0.05,
                                        max_depth=4, max_features='sqrt',
                                        min_samples_leaf=15, min_samples_split=10)
    else:
        is_classify = False
        rf = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                       max_depth=4, max_features='sqrt',
                                       min_samples_leaf=15, min_samples_split=10,
                                       loss='huber')

    rf.fit(x_train, y_train)

    print("Successfully trained model to predict: " + col_to_predict)
    print("------Evaluation-------")

    if is_classify:
        acc = accuracy_score(y_test, rf.predict(x_test))
        print('ACC         : ', round(acc, 4))
    else:
        r2 = r2_score(y_test, rf.predict(x_test))
        mse = mean_squared_error(y_test, rf.predict(x_test))
        rmse = mse ** (1 / 2)
        print('R2          : ', round(r2, 4))
        print('RMSE        : ', round(rmse, 2))

    df.loc[df[col_to_predict].isnull(), col_to_predict] = rf.predict(df_to_predict.values)


def remove_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes all redundant columns which have the same value across the column
    :param df: pandas DataFrame to remove redundant columns from
    :return: pandas DataFrame without redundant columns
    """

    return df.loc[:, df.apply(pd.Series.nunique) != 1]


def rank_features(X, Y):
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)
    forest.fit(X, Y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()
    return indices


def adjust_skewness(df: pd.DataFrame, specific: str = None) -> pd.DataFrame:
    """
    Adjusts the skewness of all columns by finding highly skewed columns
    and performing a boxcox transformation
    :param df: pandas DataFrame to adjust skewed columns in
    :return: pandas DataFrame with skew adjusted columns
    """

    numerics = list(x[0] for x in (filter(lambda x: x[1].name != 'object' and x[1].name != 'category', zip(df.columns, df.dtypes))))
    skewed_feats = df[numerics].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew': skewed_feats})
    skewness = skewness[abs(skewness) > 0.7]
    skewed_features = skewness.index
    if specific:
        skewed_features = [specific]
    lam = 0.15

    for feat in skewed_features:
        boxcot_trans = boxcox1p(df[feat], lam)
        if not boxcot_trans.isnull().any():
            df[feat] = boxcox1p(df[feat], lam)

    return df


def evaluate_regressor(y_actual: np.array, y_pred: np.array, metric_func: Callable[[np.array, np.array], float]) -> [float]:
    """
    Evaluates the prediction results of a regressor.
    :param y_actual: numpy array of actual values
    :param y_pred: numpy array of predicted values
    :param metric_func: custom function to evaluate. Default = None
    :return: Returns the evaluation metrics for the regressor.
    """

    r2 = r2_score(y_actual, y_pred)
    mse = mean_squared_error(y_actual, y_pred)
    rmse = mse ** (1 / 2)
    metrics = [r2, mse, rmse]

    print('R2          : ', round(r2, 4))
    print('MSE        : ', round(mse, 4))
    print('RMSE        : ', round(rmse, 4))

    if metric_func:
        custom_metric = metric_func(y_actual, y_pred)
        metrics.append(custom_metric)
        print('CUSTOM METRIC: ', round(houses_metric, 4))

    return metrics


class RegressorModel:

    def __init__(self, model):
        self.model = model

    def predict(self, x):
        return self.model.predict(x)


def cross_validate(model, x: np.array, y: np.array, metric: Callable[[np.array, np.array], float], folds: int = 5,
                   repeats: int = 3, verbose: bool = True) -> float:
    """
    Perform k-fold cross-validation using the out of the bag score.
    :param model: scikit-learn like model to perform cross validation on
    :param x: numpy array of features
    :param y: numpy array of predictors
    :param metric: what metric to use to evaluate the models
    :param folds: number of folds. Default = 5.
    :param repeats: number of times to repeat the whole process (different random splitting). Default = 3.
    :param verbose: Whether to print progress messages. Default = True.
    :return: the average metric score across all folds and repeats
    """

    y_pred = np.zeros(len(y))
    score = np.zeros(repeats)

    for r in range(repeats):
        if verbose:
            print('Running k-fold cross-validation ', r + 1, '/', repeats)
        x, y = shuffle(x, y, random_state=r)

        for i, (train_ind, test_ind) in enumerate(KFold(n_splits=folds, random_state=r + 10).split(x)):
            if verbose:
                print('Computing fold ', i + 1, '/', folds)
            x_train, y_train = x[train_ind, :], y[train_ind]
            x_test, y_test = x[test_ind, :], y[test_ind]
            model.fit(x_train, y_train)
            y_pred[test_ind] = model.predict(x_test)
        score[r] = metric(y_pred, y)

    print('Evaluation metric:', metric.__name__)
    print('Average:', np.round(np.mean(score), 4))
    print('Std. Dev:', np.round(np.std(score), 4))
    mean = np.mean(score)
    return mean[0] if len(mean) > 1 else mean


class EnsembleRegressor:

    def __init__(self, models):
        self.models = models

    def predict(self, x: np.array) -> np.array:
        total = np.zeros(len(x))
        for model in self.models:
            total += model.predict(x)
        total /= len(self.models)
        return total


def get_cross_validation_models(model, x: np.array, y: np.array, metric: Callable[[np.array, np.array], float], folds: int = 5,
                   repeats: int = 3, verbose: bool = True) -> EnsembleRegressor:
    """
    Perform k-fold cross-validation using the out of the bag score.
    :param model: scikit-learn like model to perform cross validation on
    :param x: numpy array of features
    :param y: numpy array of predictors
    :param metric: what metric to use to evaluate the models
    :param folds: number of folds. Default = 5.
    :param repeats: number of times to repeat the whole process (different random splitting). Default = 3.
    :param verbose: Whether to print progress messages. Default = True.
    :return: the average metric score across all folds and repeats
    """

    y_pred = np.zeros(len(y))
    score = np.zeros(repeats)
    ensemble = EnsembleRegressor([])

    for r in range(repeats):
        if verbose:
            print('Running k-fold cross-validation ', r + 1, '/', repeats)
        x, y = shuffle(x, y, random_state=r)

        for i, (train_ind, test_ind) in enumerate(KFold(n_splits=folds, random_state=r + 10).split(x)):
            if verbose:
                print('Computing fold ', i + 1, '/', folds)
            x_train, y_train = x[train_ind, :], y[train_ind]
            x_test, y_test = x[test_ind, :], y[test_ind]
            model.fit(x_train, y_train)
            ensemble.models.append(copy.copy(model))
            y_pred[test_ind] = model.predict(x_test)
        score[r] = metric(y_pred, y)

    print('Evaluation metric:', metric.__name__)
    print('Average:', np.round(np.mean(score), 4))
    print('Std. Dev:', np.round(np.std(score), 4))

    return ensemble


def try_many_regressors(x: np.array, y: np.array, metric: Callable[[np.array, np.array], float],
                        metric_max_better: bool = True) -> None:
    """
    Tries a few solid regressors in sklearn and returns the best performing one
    :param x: numpy array of the features
    :param y: numpy array of the predictor
    :param metric: Function, the evaluation metric to use.
    :param metric_max_better: If the metric's higher value means better value
    """

    gb = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state=42)

    gb2 = GradientBoostingRegressor(learning_rate=0.05, max_features='sqrt', loss='huber',
                                    min_impurity_split=None, min_samples_leaf=15,
                                    min_samples_split=10, n_estimators=12000,
                                    random_state=42)

    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=42))
    elastic = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, max_iter=10000, random_state=42))
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=42)
    rrf = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=3, random_state=42)
    huber = HuberRegressor()
    linear = LinearRegression()
    nn = MLPRegressor(hidden_layer_sizes=(1000, 10), learning_rate='adaptive',
                      max_iter=1000, random_state=42, early_stopping=True)
    svm_r = svm.SVR(kernel='poly', gamma='auto')
    knn = KNeighborsRegressor(n_neighbors=5)

    regressors = [gb, gb2, lasso, elastic, rf, rrf, huber, linear, nn, svm_r, knn]
    scores = np.zeros(len(regressors))

    for i, r in enumerate(regressors):
        print('Running k-fold cross validation for', r.__class__.__name__)
        scores[i] = cross_validate(r, x, y, metric)

    best_index = np.argmax if metric_max_better else np.argmin
    best = np.amax if metric_max_better else np.amin
    first = lambda s: s[0] if len(s) > 1 else s

    print('Best performing model: ', regressors[first(best_index(scores))].__class__.__name__)
    print('Best', metric.__name__, ':', best(scores))


def get_ensembles_many_regressors(x: np.array, y: np.array, metric: Callable[[np.array, np.array], float],
                                  metric_max_better: bool = True) -> None:
    """
    Tries a few solid regressors in sklearn and returns the best performing one
    :param x: numpy array of the features
    :param y: numpy array of the predictor
    :param metric: Function, the evaluation metric to use.
    :param metric_max_better: If the metric's higher value means better value
    """

    gb = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state=42)

    gb2 = GradientBoostingRegressor(learning_rate=0.05, max_features='sqrt', loss='huber',
                                    min_impurity_split=None, min_samples_leaf=15,
                                    min_samples_split=10, n_estimators=12000,
                                    random_state=42)

    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=42))
    elastic = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, max_iter=10000, random_state=42))
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=42)
    rrf = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=3, random_state=42)
    huber = HuberRegressor()
    linear = LinearRegression()
    nn = MLPRegressor(hidden_layer_sizes=(1000, 10), learning_rate='adaptive',
                      max_iter=1000, random_state=42, early_stopping=True)
    svm_r = svm.SVR(kernel='poly', gamma='auto')
    knn = KNeighborsRegressor(n_neighbors=5)

    regressors = [gb, gb2, lasso, elastic, rf, rrf, huber, linear, nn, svm_r, knn]
    scores = np.zeros(len(regressors))

    for i, r in enumerate(regressors):
        print('Running k-fold cross validation for', r.__class__.__name__)
        scores[i] = cross_validate(r, x, y, metric)

    best_index = np.argmax if metric_max_better else np.argmin
    best = np.amax if metric_max_better else np.amin
    first = lambda s: s[0] if len(s) > 1 else s

    print('Best performing model: ', regressors[first(best_index(scores))].__class__.__name__)
    print('Best', metric.__name__, ':', best(scores))


def evaluate_ensemble_weights(weights: np.array, model_preds: np.array, y_test: np.array,
                          metric: Callable[[np.array, np.array], float], maximize: bool = True) -> float:
    """
    Computes the metric of the ensemble according to some weights
    :param weights: the weights of each model (their priority for the prediction)
    :param model_preds: the precomputed predictions of each model (cross validated predictions)
    :param y_test: numpy.array of true values
    :param metric: metric to evaluate ensemble
    :param maximize: if a higher score means better
    :return: the computed metric of the ensemble (or negative of it)
    """

    y_pred = sum(x * y for x, y in zip(model_preds, weights))
    return -1 * metric(y_test, y_pred) if maximize else metric(y_test, y_pred)


def optimize_ensemble(ensemble: EnsembleRegressor, x_test: np.array, y_test: np.array,
                      metric: Callable[[np.array, np.array], float], metric_max_better: bool = True) -> np.array:
    """
    Optimizes the weights in the ensemble model
    :param ensemble: The ensemble containing all the models
    :param x_test: numpy array of testing features
    :param y_test: numpy array of testing predictors
    :param metric: evaluation metric to optimize
    :param metric_max_better: if higher metric score means better
    :return: The optimized weights of the ensemble model
    """

    model_preds = np.array((model.predict(x_test) for model in ensemble.models))
    x0 = np.ones(len(model_preds))
    bounds = [(0, 1)] * len(x0)
    cons = {'type': 'eq', 'fun': lambda x: sum(x) - 1}

    optimized = minimize(evaluate_ensemble_weights, x0, bounds=bounds, constraints=cons,
                         args=(model_preds, y_test, metric, metric_max_better))

    return optimized.x

