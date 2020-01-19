import os
os.chdir('..')
from eds import metrics
import numpy as np


def test_fp_rate():
    y_true = [1, 0]
    y_pred = [1, 1]
    fp_rate = metrics.false_positive_rate(np.array(y_true), np.array(y_pred))
    assert(fp_rate == 0.5)
