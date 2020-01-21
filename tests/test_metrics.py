import os
os.chdir('..')
from eds import metrics
import numpy as np


def test_fp_rate():
    y_true = [1, 0]
    y_pred = [1, 1]
    fp_rate = metrics.false_positive_rate(np.array(y_true), np.array(y_pred))
    assert(fp_rate == 0.5)

    y_true = [1, 0, 1, 1]
    y_pred = [1, 1, 1, 1]
    fp_rate = metrics.false_positive_rate(np.array(y_true), np.array(y_pred))
    assert(fp_rate == 0.25)

    y_true = [1, 1, 1, 1]
    y_pred = [1, 1, 1, 1]
    fp_rate = metrics.false_positive_rate(np.array(y_true), np.array(y_pred))
    assert(fp_rate == 0.0)


def test_fn_rate():
    y_true = [1, 0]
    y_pred = [1, 1]
    fn_rate = metrics.false_negative_rate(np.array(y_true), np.array(y_pred))
    assert(fn_rate == 1.0)

    y_true = [1, 0, 1, 1]
    y_pred = [1, 1, 0, 1]
    fn_rate = metrics.false_negative_rate(np.array(y_true), np.array(y_pred))
    assert(fn_rate == 1.0)

    y_true = [0, 0, 1, 0]
    y_pred = [0, 0, 0, 0]
    fn_rate = metrics.false_negative_rate(np.array(y_true), np.array(y_pred))
    assert(fn_rate == 0.25)

    y_true = [1, 1, 1, 1]
    y_pred = [1, 1, 1, 1]
    fn_rate = metrics.false_positive_rate(np.array(y_true), np.array(y_pred))
    assert(fn_rate == 0.0)


def test_avg_rate():
    y_true = [1, 1, 1, 1]
    y_pred = [1, 1, 1, 1]
    error = metrics.avg_error(np.array(y_true), np.array(y_pred))
    assert(error == 0.5)

    y_true = [0, 0, 0, 1]
    y_pred = [0, 0, 1, 1]
    error = metrics.avg_error(np.array(y_true), np.array(y_pred))
    assert(error == 0.25)


def test_auc_error():
    y_true = [1, 0, 0, 1]
    y_pred = [.99, 0.1, .02, .99]
    result = 1 - metrics.roc_auc_error(np.array(y_true), np.array(y_pred))
    assert(result == 1)

    y_true = [1, 0, 0, 1]
    y_pred = [.40, 0.2, .5, .32]
    result = 1 - metrics.roc_auc_error(np.array(y_true), np.array(y_pred))
    assert(result == 0.5)


def test_mse():
    y_true = [25, 0, 23, 1]
    y_pred = [26, 5, 21, 1]
    result = metrics.mse(np.array(y_true), np.array(y_pred))
    assert(result == 7.5)

    y_true = [25, 5, 23, 1]
    y_pred = [26, 5, 21, 1]
    result = metrics.mse(np.array(y_true), np.array(y_pred))
    assert(result == 1.25)


def test_r2():
    y_true = [25, 0, 23, 1]
    y_pred = [26, 5, 21, 1]
    result = round(1 - metrics.r2_error(np.array(y_true), np.array(y_pred)), 2)
    assert(result == 0.95)

    y_true = [25, 5, 21, 1]
    y_pred = [25, 5, 21, 1]
    result = round(1 - metrics.r2_error(np.array(y_true), np.array(y_pred)), 2)
    assert(result == 1)
