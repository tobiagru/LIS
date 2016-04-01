import numpy as np
import pandas as pd


# import/export functions --------------------------------------------------------------------
def load_Y(fname, usecols = 1, asNpArray = False):
     if asNpArray:
         return np.loadtxt(fname,
                           dtype = np.float32,
                           delimiter = ',',
                           skiprows = 1,
                           usecols = usecols)
     else:
         return pd.read_csv(fname,
                       index_col=0,
                       dtype=np.float32,
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
                  usecols = 0)

def write_Y(fname, Y_pred, X_test = 0, Ids = 0):
    if X_test is not 0:
        if Y_pred.shape[0] is not X_test.as_matrix().shape[0]:
            print("error - dimension of y matrix does not match number of expected predictions")
        else:
            data = pd.DataFrame(data = Y_pred, index = X_test.index, columns = ['y'])
    elif Ids is not 0:
        if Y_pred.shape[0] is not X_test.shape[0]:
            print("error - dimension of y matrix does not match number of expected predictions")
        else:
            data = pd.DataFrame(data = Y_pred, index = Ids, columns='y')
    f = open(fname, 'w+')
    data.to_csv(f)
    f.close()

def log_best_param_score(fname, date_time, clf_name, score, best_param):
    f = open(fname, 'a+')
    f.write('{0} - {1} - score: {2:.4f} - param: {3}'.format(date_time,clf_name,score,best_param))
    f.close()

def log_score(fname, date_time, clf_name, score):
    f = open(fname, 'a+')
    f.write('{0} - {1} - score: {2:.4f}'.format(date_time,clf_name,score))
    f.close()