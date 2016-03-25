#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sklearn.preprocessing as skpre


from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

from lib import *


X, Y = load_data('train')
X = skpre.StandardScaler().fit_transform(X)

lb = skpre.LabelBinarizer()
lb.fit(Y)
num_classes = len(lb.classes_)
num_features = X.shape[1]

layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('dense2', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense3', DenseLayer),
           ('output', DenseLayer)]

params = dict(
    layers=layers0,

    dropout0_p=0.5,
    dropout1_p=0.5,

    dense0_num_units=2048,
    dense1_num_units=3048,
    dense2_num_units=1000,
    dense3_num_units=200,

    input_shape=(None, num_features),
    output_num_units=num_classes,
    output_nonlinearity=softmax,

    update=nesterov_momentum,
    update_learning_rate=0.01,

    eval_size=0.2,
    verbose=1,
    max_epochs=20,
    regression=False
)

clf = NeuralNet(**params)

print '-------------------'
name = '1'
print name
print params
print '-------------------'

with Timer('testset'):
    Xvalidate, _ = load_data('validate')
    Xvalidate = skpre.StandardScaler().fit_transform(Xvalidate)
    clf.fit(X, Y)
    Yvalidate = clf.predict(Xvalidate)
    write_Y('validate_%s' % name, Yvalidate)

Xtest, _ = load_data('test')
Xtest = skpre.StandardScaler().fit_transform(Xtest)
Ytest = clf.predict(Xtest)
write_Y('test_%s' % name, Ytest)
