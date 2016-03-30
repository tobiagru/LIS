import numpy as np
import pandas as pd


def load_Y(fname, asNpArray = True):
     data = pd.read_csv(fname,
                       index_col=1,
                       dtype=np.float32,
                       header=0,
                       usecols = [1])
     if asNpArray:
         return data.as_matrix
     else:
         return data

def load_X_train(fname,asNpArray = True):
    data = pd.read_csv(fname,
                       index_col=1,
                       dtype=np.float32,
                       header=0,
                       usecols = range(2,16,1))
    if asNpArray:
        return data.as_matrix
    else:
        return data


def load_X_test(fname, asNpArray = True):
    data = pd.read_csv(fname,
                       index_col=1,
                       dtype=np.float32,
                       header=0,
                       usecols = range(1,15,1))
    if asNpArray:
        return data.as_matrix
    else:
        return data

#print pandas dataframe as np.array with dataframe.as_matrix

def write_Y(fname, Y_pred, X_test):
    if Y_pred.as_matrix.shape[1] is not X_test.as_matrix.shape[1]:
        print
    data = pd.DataFrame(data = Y_pred,
                        index = X_test.index,
                        columns = ['Id','y'])
    f = open(fname, 'w')
    data.to_csv(f)


X_train = load_X_train("/home/tg/Projects/Data/pr2/train.csv")
Y = load_Y("/home/tg/Projects/Data/pr2/train.csv")
X_test = load_X_test("/home/tg/Projects/Data/pr2/test.csv")
y_z = np.ones([X_test.shape[1],1])
write_Y("/home/tg/Projects/Data/pr2/handin_test.csv",y_z,X_test)
