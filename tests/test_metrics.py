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
