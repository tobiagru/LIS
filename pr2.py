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

#prediction functions -------------------------------------------------------------------------

from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
#from sklearn.preprocessing import StandardScaler, Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import datetime


names = ["Nearest_Neighbors", "Linear_SVM", "RBF_SVM", "Poly_SVM", "Sigmoid_SVM", "Decision_Tree",
         "Random_Forest", "AdaBoost", "Naive_Bayes", "Linear_Discriminant_Analysis",
         "Quadratic_Discriminant_Analysis"]

classifiers = [
    KNeighborsClassifier(),
    SVC(kernel="linear"),
    SVC(kernel='rbf'),
    SVC(kernel='polynomial'),
    SVC(kernel='sigmoid'),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]

param_grid = [{'n_neighbors': range(3,20,1),'leaf_size':range(20,100,5)},
              {'C': np.linspace(0,1,20)},
              {'C': np.linspace(0,1,20), 'gamma': np.linspace(0,5,20)},
              {'C': np.linspace(0,1,20), 'gamma': np.linspace(0,5,20), 'degree': range(2,4,1)},
              {'C': np.linspace(0,1,20), 'gamma': np.linspace(0,5,20)},
              {'max_depth': range(3,10,1), 'max_features': range(1,3,1)},
              {'max_depth': range(3,15,1), 'n_estimators': range(5,30,5), 'max_features': range(1,3,1)},
              {'n_estimators': range(30,150,10)},
              {},
              {},
              {}]

X_train =load_X_train("/home/tg/Projects/LIS-Data/pr2/train.csv")
X_train.shape()
y_train =load_Y("/home/tg/Projects/LIS-Data/pr2/train.csv")
X_test = load_X_test("/home/tg/Projects/LIS-Data/pr2/test.csv")

for name, clf, param in zip(names, classifiers, param_grid):
        #TODO - feature Selection
        now = datetime.datetime.now()
        date_time = '{0:02d}_{1:02d}_{2:02d}_{3:02d}_{4:02d}'.format((now.year%2000), now.month, now.day, now.hour, now.minute)
        print('\ntesting classifier {0} start at {1}\n'.format(name,date_time))

        if name is "Naive_Bayes" or "Linear_Discriminant_Analysis" or "Quadratic_Discriminant_Analysis":
            #if no gridsearch neccessary just do cross validation
            try:
                score = cross_val_score(clf, X_train.as_matrix(), y_train.as_matrix(), scoring='accuracy',cv=10)
            except:
                print('\t Problem while doing cross validation with {0}'.format(name))
                score = 0
            finally:
                clf_tmp = clf

            print('\t{0} scored {1:.4f}'.format(name,score))
            log_score("/home/tg/Projects/LIS/Data/pr2/log.txt",date_time,name,score)
        else:
            #if gridsearch necessary do it and print best solution
            try:
                grid_search = GridSearchCV(clf, param, scoring='accuracy',cv=10)
                grid_search.fit(X_train.as_matrix(), y_train.as_matrix())
            except:
                print('\t Problem while doing gridsearch with {0}\n'.format(name))
                clf_tmp = clf
                score = 0
                best_param = 0
            else:
                clf_tmp = grid_search.best_estimator_
                score = grid_search.best_score_
                best_param = grid_search.best_params_

            print('\t{0} scored {1:.4f} with param: {2}'.format(name,score,best_param))
            log_best_param_score("/home/tg/Projects/LIS/Data/pr2/log.txt",date_time,name,score,best_param)

        try:
            clf_tmp.fit(X_train.as_matrix(), y_train.as_matrix())
        except:
            print('\t Problem while training with full training set on {0}\n'.format(name))

        try:
            y_pred = clf_tmp.predict(X_test.as_matrix())
        except:
            print('\t Problem while predicting on test set with {0}\n'.format(name))
        else:
            write_Y('/home/tg/Projects/LIS/Data/pr2/handin-{0}-{1}.csv'.format(name,date_time),
                y_pred, X_test)