#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
import pandas as pd
import time


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %s' % (time.time() - self.tstart)


def load_X(fname):
    data = pd.read_csv('data/%s.csv' % fname,
                       index_col=False,
                       dtype=np.float64,
                       header=None)
    return data.as_matrix()


def load_Y(fname):
    return pd.read_csv('data/%s_y.csv' % fname,
                       index_col=False,
                       header=None).as_matrix()


def write_Y(fname, Y):
    if Y.shape[1] != 8:
        raise Exception('Y has invalid shape %s!' % str(Y.shape))
    np.savetxt('results/%s_y_pred.txt' % fname, Y, fmt='%f', delimiter=',')
