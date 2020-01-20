from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import decomposition
from eds import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def adjust_skewness(df: pd.DataFrame, specific: str = None) -> pd.DataFrame:
    """
    Adjusts the skewness of all columns by finding highly skewed columns
    and performing a boxcox transformation

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to adjust skewed columns in

    specific : str, optional
        Specfic feature to skew, by default None

    Returns
    -------
    pd.DataFrame
        Returns the pandas DataFrame with corrected columns where skewness was high.
    """

    numerics = [x[0] for x in (filter(lambda x: x[1].name != 'object' and x[1].name != 'category', zip(df.columns, df.dtypes)))]
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


def get_nan_col_proportions(df: pd.DataFrame, lowest_proportion: float = 0.0) -> [(str, float)]:
    """
    Prints out all columns with NaN values that exceed a specific proportion (default 0.0)

    Parameters
    -------
    df: pd.DataFrame
        Input DataFrame to check column NaN rates.

    lowest_proportion: float
        Threshold for unacceptable NaN rate.

    Returns
    -------
    [dict]
        List of dictionaries that contain the string of the column name and its NaN rate.
    """

    values = list(zip(list(df.isnull().columns), list(df.isnull().any())))
    filtered = list(filter(lambda x: x[1][1] == True, enumerate(values)))  # May just return x[1][1]
    contains_nan = [y for x, y in filtered]
    proportion_nan = [round(sum(df[x].isnull()) / len(df[x]), 3) for x, y in contains_nan]
    proportion_nan = [(x[0], proportion_nan[i]) for i, x in enumerate(contains_nan)]
    nan_prop_list = list()
    for col, propo_nan in proportion_nan:
        if abs(propo_nan) > lowest_proportion:
            nan_prop_list.append({'Column_Name': col, 'NaN_Proportion': propo_nan})
    return nan_prop_list


def remove_nan_cols(df: pd.DataFrame, prop_threshold: float = 0.0) -> pd.DataFrame:
    """
    Removes columns from a DataFrame that have an unacceptable NaN rate.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check column NaN rates.

    prop_threshold : float, optional
        Threshold for unacceptable NaN rate, by default 0.0

    Returns
    -------
    pd.DataFrame
        DataFrame that removed the unacceptable NaN columns.
    """

    nan_props = get_nan_col_proportions(df, prop_threshold)
    names = [n['Column_Name'] for n in nan_props]
    df.drop(columns=names, inplace=True)
    return df


def get_moderate_correlations(df: pd.DataFrame, col_to_correlate: str, moderate_value: float = 0.4) -> None:
    """
    Prints out all correlations deemed as moderate (0.4, or set by parameter).

    Parameters
    ----------
    df : pd.DataFrame
        pandas DataFrame to check correlations from.

    col_to_correlate : str
        String that represents column we want to check correlations with.

    moderate_value : float, optional
        Which correlation value we deem as moderate (default 0.4)., by default 0.4

    Returns
    -------
    [dict]
        List of dictionaries that contain the column name and its correlation.
    """

    result = []
    if df[col_to_correlate].dtype.name == 'category':
        df[col_to_correlate] = df[col_to_correlate].cat.codes
    corrs = df[df.columns].corr()
    cols = corrs.columns
    corrs_value = corrs[col_to_correlate]
    for col, corr_value in zip(cols, corrs_value):
        if abs(corr_value) > moderate_value and col != col_to_correlate:
            result.append({'Column_Name': col, 'Correlation:': round(corr_value, 2)})
    return result


def remove_weak_correlations(df: pd.DataFrame, target_col: str, weak_threshold: float = 0.1) -> pd.DataFrame:
    """
    Removes weak correlations from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to which remove weak correlations from.

    target_col : str
        Target column to check correlations with.

    weak_threshold : float, optional
        Threshold to consider a weak correlation, by default 0.1

    Returns
    -------
    pd.DataFrame
        DataFrame without weak correlations.
    """

    cols = df.corr().columns
    corrs = df.corr()[target_col]
    strongly_correlated = list()
    for col, corr in zip(cols, corrs):
        if abs(corr) >= weak_threshold:
            strongly_correlated.append(col)
    for col in df.columns:
        if col not in strongly_correlated:
            df.drop(columns=[col], inplace=True)
    return df


def remove_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes redundant constant columns from a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to return without constant columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame without constant columns.

    """

    return df.loc[:, df.apply(pd.Series.nunique) != 1]


def convert_categorical_to_numbers(to_change_df: pd.DataFrame) -> pd.DataFrame:
    """
    Dummifies all category data including objects.
    Assumes the data has been cleaned and the dtypes are consistent.

    Parameters
    ----------
    to_change_df : pd.DataFrame
        pandas DataFrame to convert to all numerical

    Returns
    -------
    pd.DataFrame
        Dummified input pandas DataFrame
    """
    return pd.get_dummies(convert_objects_to_categories(to_change_df))


def convert_objects_to_categories(to_change_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts all object dtypes into categories

    Parameters
    ----------
    to_change_df : pd.DataFrame
        DataFrame to convert objects to categories

    Returns
    -------
    pd.DataFrame
        DataFrame with categories instead of objects
    """

    for col, dtype in zip(to_change_df.columns, to_change_df.dtypes):
        if dtype == object:
            to_change_df[col] = to_change_df[col].astype('category')
    return to_change_df


def replace_missing_with_ml(df: pd.DataFrame, col_to_predict: str) -> (pd.DataFrame, dict):
    """
    Replace the missing values in the given column using machine learning predictions

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to use as features (and predictor column)

    col_to_predict : str
        string that represents the predictor column

    Returns
    -------
    (pd.DataFrame, dict)
        DataFrame with filled predictor column values via machine learning
        Also returns a dictionary containing performance information.

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

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1337)

    is_classify = True

    if df[col_to_predict].dtype.name == 'object':
        rf = ensemble.GradientBoostingClassifier()
    else:
        is_classify = False
        rf = ensemble.GradientBoostingRegressor()

    rf.fit(x_train, y_train)

    results = {}

    if is_classify:
        y_hat = rf.predict(x_test)
        roc_auc = 1 - metrics.roc_auc_error(y_test, rf.predict_proba(x_test))
        fp_rate = metrics.fp_rate(y_test, y_hat)
        fn_rate = metrics.fn_rate(y_test, y_hat)
        results['ROC AUC'] = roc_auc
        results['False Positive Rate'] = fp_rate
        results['False Negative Rate'] = fn_rate
    else:
        r2 = metrics.r2_score(y_test, rf.predict(x_test))
        mse = metrics.mean_squared_error(y_test, rf.predict(x_test))
        rmse = mse ** (1 / 2)
        results['R2'] = r2
        results['Mean Squared Error'] = mse
        results['Root Mean Squared Error'] = rmse

    df.loc[df[col_to_predict].isnull(), col_to_predict] = rf.predict(df_to_predict.values)

    return df, results


def apply_scale(x: np.array, scale_type: str = 'Standard') -> np.array:
    """
    Scales the data according to the distribution of x_train

    Parameters
    ----------
    x : np.array
        numpy.array to scale via its distribution

    scale_type : str, optional
        Which scaling type to use. Options: 'Standard', 'Robust', 'MinMax'. Default = 'Standard'

    Returns
    -------
    np.array
        x properly scaled.

    """

    if scale_type == 'Standard':
        scaler = preprocessing.StandardScaler()
    elif scale_type == 'Robust':
        scaler = preprocessing.RobustScaler()
    elif scale_type == 'MinMax':
        scaler = preprocessing.MinMaxScaler()
    else:
        raise Exception('Invalid string input for scale_type')

    scaler.fit(x)  # Correct to use only the training data to not bias our model's test evaluation
    return scaler.transform(x)


def apply_pca(x: np.array, n_comps: float = 0.975) -> np.array:
    """
    Apply PCA to the data according to the distribution of x_train

    Parameters
    ----------
    x : np.array
        numpy.array to scale via its distribution

    n_comps : float, optional
        Either the number of principal components, or the variance to be kept in the components., by default 0.975

    Returns
    -------
    np.array
        X set of feature vectors with PCA applied.
    """

    pca = decomposition.PCA(n_components=n_comps)
    pca.fit(x)  # Correct to use only the training data to not bias our model's test evaluation
    return pca.transform(x)
