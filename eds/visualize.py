import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_hist_distribution(df: pd.Series, col: str) -> None:
    """
    Plots a histogram of the column's distribution
    :param df: pandas DataFrame to extract column from
    :param col: string that represents column
    """

    df[col].hist(bins=50, facecolor='red', alpha=0.5)
    plt.title('Distribution of ' + col, fontsize=15)
    plt.xlabel(col, fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    plt.show()


def plot_relationship(df: pd.DataFrame, feature1: str, feature2: str, fit_line=True) -> None:
    """
    Plots the relationship of 2 features to each other
    :param df: pandas DataFrame where features are contained
    :param feature1: feature 1's column name
    :param feature2: feature 2's column name
    """
    if fit_line:
        plt.plot(np.unique(df[feature1]), np.poly1d(np.polyfit(df[feature1], df[feature2], 1))(np.unique(df[feature1])), color='red')
    plt.scatter(df[feature1], df[feature2], s=50, color='blue')
    plt.grid()
    plt.title('Pairwise Feature Relationship', fontsize=15)
    plt.xlabel(feature1, fontsize=20)
    plt.ylabel(feature2, fontsize=20)
    axes = plt.gca()
    x_pad = (df[feature1].max() - df[feature1].min()) * 0.05
    y_pad = (df[feature2].max() - df[feature2].min()) * 0.05
    axes.set_xlim([df[feature1].min() - x_pad, df[feature1].max() + x_pad])
    axes.set_ylim([df[feature2].min() - y_pad, df[feature2].max() + y_pad])
    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    plt.show()


def plot_binary_feature_distribution(df, feature, class_name, hist=False):
    """
    Plots 2 distributions of a feature: class_name or not class_name
    This helps us to see where the difference in distributions
    :param df: pandas DataFrame to
    :param feature: Which feature to check the distributions of
    :param class_name: The class we draw an independent distribution of
    :param is_hist: Plot a histogram or KDE. Default = False
    """

    bins = np.linspace(df[feature].min(), df[feature].max(), 30)

    if hist:
        sns.distplot([df[df[class_name] == 0][feature], df[df[class_name] == 1][feature]], hist=True,
                     norm_hist=True, kde=False,
                     bins=bins, color=['blue', 'red'], label=['True', 'False'],
                     hist_kws={'edgecolor': 'black'})
    else:
        sns.kdeplot(df[df[class_name] == 1][feature], shade=True, color='Blue', label='True')
        sns.kdeplot(df[df[class_name] == 0][feature], shade=True, color='Red', label='False')

    plt.legend(prop={'size': 13}, title=class_name)
    plt.title('Density Plot of ' + feature, fontsize=15)
    plt.xlabel(feature, fontsize=20)
    plt.ylabel('Density', fontsize=20)
    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    plt.show()