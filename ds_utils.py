from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np


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
    proportion_nan = [sum(df[x].isnull()) / len(df[x]) for x, y in contains_nan]
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
    :return: None
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
    :return: None
    """

    if df[col_to_correlate].dtype.name == 'category':
        df[col_to_correlate] = df[col_to_correlate].cat.codes
    corrs = df[df.columns].corr()
    cols = corrs.columns
    corrs_value = corrs[col_to_correlate]
    for col, corr_value in zip(cols, corrs_value):
        if abs(corr_value) > moderate_value and col != col_to_correlate:
            print(col, ': ', corr_value)


def remove_weak_correlations(df: pd.DataFrame, col_to_correlate: str, weak_threshold: float = 0.05) -> pd.DataFrame:
    """
    Removes weak correlations
    :param df: pandas DataFrame to remove columns from.
    :param col_to_correlate: String column name to check correlation with
    :param weak_threshold: float number that counts as an absolute weak threshold
    :return: pandas DataFrame without the columns weakly correlated to target
    """
    cols = df[df.columns].corr().columns
    corrs = df[df.columns].corr()[col_to_correlate]
    weakly_correlated = list()
    for col, corr in zip(cols, corrs):
        if abs(corr) < weak_threshold and col != col_to_correlate:
            weakly_correlated.append(col)
    return df.drop(columns=weakly_correlated)


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


def replace_missing_with_ml(df: pd.DataFrame, col_to_predict: str) -> pd.DataFrame:
    """
    Replace the missing values in the given column using machine learning predictions
    :param df: pandas DataFrame to use as features (and predictor column)
    :param col_to_predict: string that represents the predictor column
    :return: pandas DataFrame with filled predictor column values via machine learning
    """
    y = df[col_to_predict].values

    dummified_df = df.copy()

    cols_to_drop = filter(lambda t: t[1], zip(df.columns, df.isnull().any()))
    dummified_df = dummified_df.drop(columns=cols_to_drop)
    dummified_df = convert_categorical_to_numbers(dummified_df)

    dummified_df[col_to_predict] = y

    df_to_model = dummified_df[dummified_df[not col_to_predict].isnull()]

    df_to_predict = dummified_df[dummified_df[col_to_predict].isnull()]
    df_to_predict = df_to_predict.drop(columns=[col_to_predict])

    y = df_to_model[col_to_predict]
    x = df_to_model.drop(columns=[col_to_predict]).values

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

    if df[col_to_predict].dtype.name == 'category':
        rf = GradientBoostingClassifier(
                                        learning_rate=0.05, max_features='sqrt',
                                        min_impurity_split=None, min_samples_leaf=15,
                                        min_samples_split=10, n_estimators=12000)
    else:
        rf = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                       max_depth=4, max_features='sqrt',
                                       min_samples_leaf=15, min_samples_split=10,
                                       loss='huber')

    rf.fit(x_train, y_train)

    print("Successfully trained model to predict: " + col_to_predict)
    print("------Evaluation-------")

    if not is_classify:
        r2 = r2_score(y_test, rf.predict(x_test))
        mse = mean_squared_error(y_test, rf.predict(x_test))
        rmse = mse ** (1 / 2)
        print('R2          : ', round(r2, 4))
        print('RMSE        : ', round(rmse, 2))
    else:
        print('ACC         : ', round(rf.score(x_test, y_test), 4))

    df.loc[df[col_to_predict].isnull(), col_to_predict] = rf.predict(df_to_predict.values)
    return df


def remove_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes all redundant columns which have the same value across the column
    :param df: pandas DataFrame to remove redundant columns from
    :return: pandas DataFrame without redundant columns
    """

    return df.loc[:, df.apply(pd.Series.nunique) != 1]


def adjust_skewness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjusts the skewness of all columns by finding highly skewed columns
    and performing a boxcox transformation
    :param df: pandas DataFrame to adjust skewed columns in
    :return: pandas DataFrame with skew adjusted columns
    """

    numerics = filter(lambda x: x[1].name != 'object' and x[1].name != 'category', zip(df.columns, df.dtypes))
    skewed_feats = df[numerics].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew': skewed_feats})
    skewness = skewness[abs(skewness) > 0.7]
    skewed_features = skewness.index
    lam = 0.15

    for feat in skewed_features:
        boxcot_trans = boxcox1p(df[feat], lam)
        if not boxcot_trans.isnull().any():
            df[feat] = boxcox1p(df[feat], lam)

    return df
