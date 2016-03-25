# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
import datetime as dt
import dateutil

from sklearn.linear_model import Ridge
import sklearn.cross_validation as skcv
import sklearn.metrics as skmet


def load_data(fname):
    data = pd.read_csv('data/%s.csv' % fname,
                       index_col=False,
                       header=None,
                       names=['date', 'A', 'B', 'C', 'D', 'E', 'F'])
                     
    data['date'] = data['date'].apply(dateutil.parser.parse)    
    data['weekday'] = data['date'].apply(dt.datetime.weekday)
    data['weekend'] = data['date'].apply(dt.datetime.weekday) >4
    data['notWeekend'] =  (data['weekend'] *-1)+1
    data['month'] = data['date'].apply(lambda x:x.month)
    data['hour'] = data['date'].apply(lambda x:x.hour)
    data['year'] = data['date'].apply(lambda x:x.year)
    data['dayofyear'] = data['date'].apply(lambda x:x.dayofyear)
       
    
    del data['date']
    return data


def apply_polynominals(X, column, p=30):
    for i in range(2, p + 1):
        X['%s^%d' % (column, i)] = np.power(X[column], i)

def apply_mult(X, column1, column2, p=0):
    X['%s_mul_%s' % (column1,column2)] = \
        X[column1] * X[column2]
    if (p>0):
        apply_polynominals(X, '%s_mul_%s' % (column1,column2),p )
    
def transformFeatures(X):
    
    X['B0'] = X['B'] == 0
    X['B1'] = X['B'] == 1
    X['B2'] = X['B'] == 2
    X['B3'] = X['B'] == 3    
    del X['B']
        
    apply_polynominals(X, 'A', 5)
    apply_polynominals(X, 'D', 5)
    apply_polynominals(X, 'E', 5)
    apply_polynominals(X, 'F', 5)
    apply_polynominals(X, 'C', 5)
    apply_polynominals(X, 'hour', 5)
    apply_polynominals(X, 'year', 3)
    apply_polynominals(X, 'dayofyear', 5)
    
    apply_mult(X, 'hour', 'A', 2)
    apply_mult(X, 'hour', 'B0', 4)
    apply_mult(X, 'hour', 'B1', 4)
    apply_mult(X, 'hour', 'B2', 4)
    apply_mult(X, 'hour', 'B3', 4)
    apply_mult(X, 'hour', 'C', 2)
    apply_mult(X, 'hour', 'D', 2)
    apply_mult(X, 'hour', 'E', 2)
    apply_mult(X, 'hour', 'F', 2)
    apply_mult(X, 'hour', 'weekend', 9)
    apply_mult(X, 'hour', 'notWeekend', 9)
    apply_mult(X, 'hour', 'weekday', 1)    

    return X

    
def logscore(gtruth, pred):
    gtruth=np.array(gtruth,dtype=float)
    pred=np.array(pred,dtype=float)
    pred = np.clip(pred, 0, np.inf)
    logdif = np.log(1 + gtruth) - np.log(1 + pred)
    return np.sqrt(np.mean(np.square(logdif)))

 
def plot(X):
    import matplotlib.pylab as plt    
    pltData =  np.hstack ( (np.log(Y),X) ) 
    pltData= pltData[pltData[:,0].argsort()]
    plt.plot(pltData,'.')
    plt.show()
 
 
def reg_lin():
    Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=.8)
    
    regressor = Ridge(alpha=1)  
    regressor.fit(Xtrain,np.log(Ytrain))
    
    Ypred = np.array(regressor.predict(Xtest),dtype=float) 
    
    print logscore( Ytest, np.exp(Ypred ) )
        
    validate = load_data('validate')
    validate = transformFeatures(validate)
    np.savetxt('results/validate.txt', np.exp(np.array( regressor.predict(validate), dtype=np.dtype('d'))))
    
def reg_crossval():
    # cross-validation of the same regressor
    regressor = Ridge()
    scorefun = skmet.make_scorer(logscore)
    scores = skcv.cross_val_score(regressor, X, Y, scoring=scorefun, cv=5)
    print 'C-V score =', np.mean(scores), '+/-', np.std(scores)


X = load_data('train')
X = transformFeatures(X)
Y = pd.read_csv('data/train_y.csv',
    index_col=False,
    header=None,
    names=['y'])
reg_lin()
