from sklearn import ensemble
from sklearn import svm
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline


def get_default_classfiers():
    """
    Gets a list of classifiers that generally perform decently.
    Note: For the ensemble class, adding more classifiers
    introduces more complexity.

    Returns
    -------
    []
        List of default classifiers.
    """
    return [
        KNeighborsClassifier(3),
        KNeighborsClassifier(7),
        make_pipeline(preprocessing.StandardScaler(), LogisticRegression(solver='lbfgs', tol=1e-2, 
                      max_iter=200, random_state=1337)),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        RandomForestClassifier(n_estimators=100, min_samples_leaf=3, random_state=1337),
        GradientBoostingClassifier(n_estimators=100, learning_rate=0.25, max_depth=6,
                                   max_features='sqrt', subsample=0.8, random_state=1337),
        AdaBoostClassifier(random_state=1337),
        GaussianNB(),
    ]


def get_default_classifiers_names():
    """
    Gets a list of the classifiers names from the above classifiers.
    Will be combined with above function in future releases.

    Returns
    -------
    []
        List of default classifiers' names.
    """
    return [
        '3-Nearest Neighbors',
        '7-NearestNeighbors',
        'Logistic Regression',
        'Random Forest',
        'Random Forest (more trees)',
        'Gradient Boosting',
        'Ada Boosting',
        'Naive Bayes',
    ]


def get_default_regressors():
    """
    Gets a list of regressors that generally perform decently.
    Note: For the ensemble class, adding more regressors
    introduces more complexity.

    Returns
    -------
    []
        List of default regressors.
    """
    return [
        make_pipeline(preprocessing.RobustScaler(), linear_model.Lasso(alpha=0.0005, random_state=1337)),
        make_pipeline(preprocessing.RobustScaler(), linear_model.ElasticNet(alpha=0.0005, l1_ratio=.9,
                                                                            max_iter=10000, random_state=1337)),
        ensemble.RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=1337),
        ensemble.ExtraTreesRegressor(n_estimators=200, min_samples_leaf=3, random_state=1337),
        ensemble.GradientBoostingRegressor(n_estimators=500, learning_rate=0.1,
                                           max_depth=4, max_features='sqrt',
                                           min_samples_leaf=15, min_samples_split=10,
                                           loss='huber', random_state=1337),
        ensemble.GradientBoostingRegressor(),
        linear_model.HuberRegressor(),
        linear_model.LinearRegression(),
        make_pipeline(preprocessing.StandardScaler(), svm.SVR(kernel='poly', gamma='auto')),
        KNeighborsRegressor(n_neighbors=5)
    ]


def get_default_regressors_names():
    """
    Gets a list of the regressors names from the above regressors.
    Will be combined with above function in future releases.

    Returns
    -------
    []
        List of default regressors' names.
    """
    return [
        'Lasso',
        'Elastic Net',
        'Random Forest',
        'Extremely Random Forest',
        'Gradient Boosting 1',
        'Gradient Boosting 2',
        'Huber Rehressor',
        'Support Vector Machine',
        '5-Nearest-Neighbors'
    ]
