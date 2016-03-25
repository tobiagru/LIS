#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sklearn.preprocessing as skpre
import sklearn.semi_supervised as sksemi

from lib import *


def preprocess_features(X):
    X = skpre.StandardScaler().fit_transform(X)


def load_data():
    global Xtrain, Ytrain, Xtest, Xvalidate
    Xtrain, Ytrain = load_X('train'), load_Y('train')
    print 'Xtrain, Ytrain:', Xtrain.shape, Ytrain.shape

    Xtest, Xvalidate = load_X('test'), load_X('validate')
    print 'Xtest, Xvalidate:', Xtest.shape, Xvalidate.shape

    preprocess_features(Xtrain)
    preprocess_features(Xtest)
    preprocess_features(Xvalidate)

load_data()
clf = sksemi.LabelSpreading(kernel='rbf', gamma=0.4, max_iter=9)
clf.fit(Xtrain, Ytrain.ravel())

Ypred = clf.predict_proba(Xvalidate)
write_Y('validate', np.nan_to_num(Ypred))
YpredTest = clf.predict_proba(Xtest)
write_Y('test', np.nan_to_num(YpredTest))
