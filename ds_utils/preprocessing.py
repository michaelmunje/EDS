import sklearn
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
        scaler = sklearn.preprocessing.StandardScaler()
    elif scale_type == 'Robust':
        scaler = sklearn.preprocessing.RobustScaler()
    elif scale_type == 'MinMax':
        scaler = sklearn.preprocessing.MinMaxScaler()
    else:
        raise Exception('Invalid string input for scale_type')

    scaler.fit(x_train) # Correct to use only the training data to not bias our model's test evaluation 
    return scaler.transform(x_train), scaler.transform(x_test)


def apply_pca(x_train: np.array, x_test: np.array, n_comps: float) -> (np.array, np.array):
    """
    Apply PCA to the data according to the distribution of x_train
    :param x_train: numpy.array to scale via its distribution
    :param x_test: numpy.array to scale according to x_test's distribution
    :param n_comps: Either the number of principal components, or the variance to be kept in the components.
    :return: x_train and x_test with PCA applied.
    """

    pca = sklearn.decomposition.PCA(n_components=n_comps)
    pca.fit(x_train) # Correct to use only the training data to not bias our model's test evaluation
    return pca.transform(x_train), pca.transform(x_test)