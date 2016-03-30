import numpy as np
import pandas as pd
import sklearn as sl

# import/export functions --------------------------------------------------------------------
def load_Y(fname, asNpArray = False):
     data = pd.read_csv(fname,
                       index_col=0,
                       dtype=np.float32,
                       header=0,
                       usecols = [0,1])
     if asNpArray:
         return data.as_matrix
     else:
         return data

def load_X_train(fname,asNpArray = False):
    data = pd.read_csv(fname,
                       index_col=0,
                       dtype=np.float32,
                       header=0,
                       usecols = [0] + list(range(2,17,1)))
    if asNpArray:
        return data.as_matrix
    else:
        return data


def load_X_test(fname, asNpArray = False):
    data = pd.read_csv(fname,
                       index_col=0,
                       dtype=np.float32,
                       header=0,
                       usecols = range(0,16,1))
    if asNpArray:
        return data.as_matrix
    else:
        return data

def create_Y(Y_pred,X_test):
    try:
        Y_pred.shape[0] is not X_test.as_matrix().shape[0]
    except:
        print("error - dimension of y matrix does not match number of expected predictions")
    else:
        return pd.DataFrame(data = Y_pred, index = X_test.index, columns = ['y'])

def write_Y(fname, Y_pred, X_test = 0):
    if X_test is not 0:
        data = create_Y(Y_pred,X_test)
    f = open(fname, 'w')
    data.to_csv(f)

#prediction functions -------------------------------------------------------------------------


#TODO - Scorefunction

#TODO - Crossval
#TODO - feature Selection
#TODO - pipeline

#TODO - SVM (RBF, Gaussian, Linear)

#TODO - DecisionTrees

#TODO - NearestNeighbour

#TODO - Stochastic Gradient Decent


