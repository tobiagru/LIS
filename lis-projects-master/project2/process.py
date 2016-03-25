#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pandas as pd
import sklearn.ensemble as skens
import sklearn.preprocessing as skpre

from lib import *


def preprocess_features(X):
    del X['B']


class UseY1Classifier(object):
    threshold_y1 = "0.08*mean"
    threshold_y2 = "0.06*mean"
    n_est = 200

    def __init__(self):
        # we need 2 separate classifiers
        self.clf1 = skens.RandomForestClassifier(n_estimators=self.n_est)
        self.clf2 = skens.RandomForestClassifier(n_estimators=self.n_est)

        # random forests to throw out unimportant features
        self.trsf1 = skens.RandomForestClassifier(n_estimators=self.n_est)
        self.trsf2 = skens.RandomForestClassifier(n_estimators=self.n_est)

    def get_wgth(self, y):
        e_count = np.bincount(y)
        if e_count.shape[0] > 4:
            e_count = e_count[1:8]
        up = np.atleast_2d(np.max(e_count) / e_count)
        weights = up[0, y - 1]
        return weights

    def _binarize(self, y):
        y = np.atleast_2d(y).T
        enc = skpre.OneHotEncoder(sparse=False)
        enc.fit(y)
        return enc.transform(y)

    def fit(self, X, Y):
        if isinstance(Y, pd.DataFrame):
            # make a numpy ndarray out of pandas dataframe
            Y = Y.as_matrix()
            X = X.as_matrix()
        # append y1 to X
        X_y1 = np.concatenate([X, self._binarize(Y[:, 0])], axis=1)

        # normalize
        X_y1 = skpre.StandardScaler().fit_transform(X_y1)
        X = skpre.StandardScaler().fit_transform(X)

        # reduce number of features
        self.trsf1.fit(X, Y[:, 0])
        self.trsf2.fit(X_y1, Y[:, 1])
        X_for_y1 = self.trsf1.transform(X, threshold=self.threshold_y1)
        X_for_y2 = self.trsf2.transform(X_y1, threshold=self.threshold_y2)

        # fit X vs y1
        self.clf1.fit(X_for_y1, Y[:, 0], sample_weight=self.get_wgth(Y[:, 0]))
        # fit X + y1 vs y2
        self.clf2.fit(X_for_y2, Y[:, 1], sample_weight=self.get_wgth(Y[:, 1]))
        return self

    def predict(self, X):
        # normalize, reduce number of features
        X = skpre.StandardScaler().fit_transform(X)
        X_for_y1 = self.trsf1.transform(X, threshold=self.threshold_y1)

        # pred y1 from X
        y1 = self.clf1.predict(X_for_y1)

        # add y1 to X, reduce number of features, normalize
        X_y1 = np.concatenate([X, self._binarize(y1)], axis=1)
        X_for_y2 = skpre.StandardScaler().fit_transform(X_y1)
        X_for_y2 = self.trsf2.transform(X_for_y2, threshold=self.threshold_y2)
        # pred y2 from X + y1
        y2 = self.clf2.predict(X_for_y2)
        return np.vstack([y1, y2]).T

    def get_params(self, *x, **xx):
        return {}


def testset_validate(clf):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split_pd(X, Y, train_size=.8)
    clf.fit(Xtrain, Ytrain)
    Ypred = clf.predict(Xtest)
    sc = score(Ytest, Ypred)
    print 'Testset score = %.4f Grade = %d%%' % (sc, grade(sc))


def predict_validation_set(clf):
    clf.fit(X, Y)
    Xvalidate = load_X('validate')
    preprocess_features(Xvalidate)
    Yvalidate = clf.predict(Xvalidate)
    write_Y('validate', Yvalidate)

X, Y = load_X('train'), load_Y('train')
preprocess_features(X)

clf = UseY1Classifier()
testset_validate(clf)
predict_validation_set(clf)
