#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pandas as pd
import numpy as np
import sklearn.cross_validation as skcv


def train_test_split_pd(X, Y, train_size):
    """
    Like sklearn.cross_validation.train_test_split but retains
    pandas DataFrame column indizes.
    """
    Xtrain, Xtest, Ytrain, Ytest = \
        skcv.train_test_split(X, Y, train_size=train_size)

    return (
        pd.DataFrame(Xtrain, columns=X.columns),
        pd.DataFrame(Xtest, columns=X.columns),
        pd.DataFrame(Ytrain, columns=Y.columns),
        pd.DataFrame(Ytest, columns=Y.columns),
    )


def load_X(fname):
    names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    names += ['K%02d' % i for i in range(1, 4 + 1)]
    names += ['L%02d' % i for i in range(1, 40 + 1)]
    data = pd.read_csv('data/%s.csv' % fname,
                       index_col=False,
                       dtype=np.float64,
                       header=None,
                       names=names)
    return data


def load_Y(fname):
    return pd.read_csv('data/%s_y.csv' % fname,
                       index_col=False,
                       header=None,
                       names=['y1', 'y2'])


def write_Y(fname, Y):
    if Y.shape[1] != 2:
        raise 'Y has invalid shape!'
    np.savetxt('results/%s_y_pred.txt' % fname, Y, fmt='%d', delimiter=',')


def score(Ytruth, Ypred):
    if Ytruth.shape[1] != 2:
        raise 'Ytruth has invalid shape!'
    if Ypred.shape[1] != 2:
        raise 'Ypred has invalid shape!'

    sum = (Ytruth != Ypred).astype(float).sum().sum()
    return sum / np.product(Ytruth.shape)


def grade(score):
    BE = 0.3091365975175955
    BH = 0.1568001421417719
    if score > BE:
        return 0
    elif score <= BH:
        return 100
    else:
        return (1 - (score - BH) / (BE - BH)) * 50 + 50
