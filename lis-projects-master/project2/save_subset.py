#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
Loads the training data, and saves a subset of it to an
excel-readable CSV-file (data/subset.csv).
"""

import numpy as np

from lib import *


X = load_X('train')
Y = load_Y('train')

X, _, Y, _ = train_test_split_pd(X, Y, train_size=.1)

X['y1'] = Y['y1']
X['y2'] = Y['y2']

np.savetxt('data/subset.csv', X,
           header=';'.join(X.columns),
           comments='',
           fmt='%.5f',
           delimiter=';')
