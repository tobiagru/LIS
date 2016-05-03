import numpy as np
import pandas as pd
import h5py
import sys
import traceback


# import/export functions --------------------------------------------------------------------
def load_Y(fname, usecols = [1], asNpArray = False):
     if asNpArray:
         return np.loadtxt(fname,
                           dtype = np.int32,
                           delimiter = ',',
                           skiprows = 1,
                           usecols = usecols)
     else:
         return pd.read_csv(fname,
                       index_col=0,
                       dtype=np.int32,
                       header=0,
                       usecols = [0] + list(usecols))

def load_X_train(fname, usecols = range(2,17,1), asNpArray = False):
    if asNpArray:
        return np.loadtxt(fname,
                          dtype = np.float32,
                          delimiter = ',',
                          skiprows = 1,
                          usecols = list(usecols))
    else:
        return pd.read_csv(fname,
                       index_col=0,
                       dtype=np.int16,
                       header=0,
                       usecols = [0] + list(usecols))


def load_X_test(fname, usecols = range(1,16,1), asNpArray = False):
    if asNpArray:
        return np.loadtxt(fname,
                          dtype = np.float32,
                          delimiter = ',',
                          skiprows = 1,
                          usecols = list(usecols))
    else:
        return pd.read_csv(fname,
                       index_col=0,
                       dtype=np.float32,
                       header=0,
                       usecols = list(usecols))

def load_Ids_test(fname):
    return np.loadtxt(fname,
                   dtype = np.float32,
                   delimiter = ',',
                   skiprows = 1,
                   usecols = [0])

def load_h5_train(fname):
    f = h5py.File(fname, 'r+')
    ids = np.zeros(f["train/axis1"].shape, dtype=np.int32)
    f["train/axis1"].read_direct(ids)
    X_train = np.zeros(f["train/axis1"].shape, dtype=np.int32)
    f["train/axis1"].read_direct(ids)
    y_train = np.zeros(f["train/axis1"].shape, dtype=np.int32)
    f["train/axis1"].read_direct(ids)



def write_Y(fname, Y_pred, X_test = 0, Ids = 0):
    if X_test is not 0:
        if Y_pred.shape[0] != X_test.as_matrix().shape[0]:
            print("error - dimension of y matrix does not match number of expected predictions")
            print('y: {0} - expected: {1}'.format(Y_pred.shape,X_test.as_matrix().shape))
        else:
            data = pd.DataFrame(data = Y_pred, index = X_test.index, columns = ['y'])
            f = open(fname, 'w+')
            data.to_csv(f)
            f.close()
    elif Ids is not 0:
        if Y_pred.shape[0] != Ids.shape[0]:
            print("error - dimension of y matrix does not match number of expected predictions")
            print('y: {0} - expected: {1}'.format(Y_pred.shape,Ids.shape))
        else:
            np.savetxt(fname=fname,X= np.column_stack([Ids,Y_pred]),
                       fmt=['%d', '%d'],delimiter=',',header='Id,y',comments='')


def log_best_param_score(fname, date_time, clf_name, score, best_param):
    f = open(fname, 'a+')
    f.write('{0} - {1} - score: {2:.4f} - param: {3}\n'.format(date_time,clf_name,score,best_param))
    f.close()

def log_score(fname, date_time, clf_name, score):
    f = open(fname, 'a+')
    f.write('{0} - {1} - score: {2:.4f}\n'.format(date_time,clf_name,score))
    f.close()