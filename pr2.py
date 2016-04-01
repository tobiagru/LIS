#prediction functions -------------------------------------------------------------------------
import numpy as np
import pandas as pd
import sys, traceback
import lib_IO
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import datetime


names = ["Nearest_Neighbors",
         #"Linear_SVM",
         #"RBF_SVM",
         #"Poly_SVM",
         #"Sigmoid_SVM",
         #"Decision_Tree",
         "Random_Forest",
         #"AdaBoost",
         #"Naive_Bayes",
         #"Linear_Discriminant_Analysis",
         #"Quadratic_Discriminant_Analysis"
        ]

classifiers = [
    KNeighborsClassifier(),
    #SVC(kernel="linear"),
    #SVC(kernel='rbf'),
    #SVC(kernel='poly'),
    #SVC(kernel='sigmoid'),
    #DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    #AdaBoostClassifier(),
    #GaussianNB(),
    #LinearDiscriminantAnalysis(),
    #QuadraticDiscriminantAnalysis()
    ]

param_grid = [{'n_neighbors': np.arange(3,20,dtype=np.int16),'leaf_size':np.arange(7,15,dtype=np.int16)},
              #{'C': np.linspace(0.05,1,20)},
              #{'C': np.linspace(0.05,1,20), 'gamma': np.linspace(0.05,5,20)},
              #{'C': np.linspace(0.05,1,20), 'gamma': np.linspace(0.05,5,20), 'degree': np.arange(2,4,dtype=np.int16)},
              #{'C': np.linspace(0.05,1,20), 'gamma': np.linspace(0.05,5,20)},
              #{'max_depth': np.arange(3,10,dtype=np.int16), 'max_features': np.arange(1,3,dtype=np.int16)},
              {'max_depth': np.arange(3,15,dtype=np.int16), 'n_estimators': np.arange(10,400,10,dtype=np.int16), 'max_features': np.arange(1,3,dtype=np.int16)},
              #{'n_estimators': np.arange(30,150,10,dtype=np.int16)},
              #{},
              #{},
              #{}
              ]


X_train =lib_IO.load_X_train("/home/tg/Projects/LIS-Data/pr2/train.csv",usecols=range(2,17,1),asNpArray=True)
print('loaded X_train with shape: {0}'.format(X_train.shape))
y_train =lib_IO.load_Y("/home/tg/Projects/LIS-Data/pr2/train.csv", usecols=[1],asNpArray=True)
print('loaded y_train with shape: {0}'.format(y_train.shape))
X_test = lib_IO.load_X_test("/home/tg/Projects/LIS-Data/pr2/test.csv", usecols=range(1,16,1), asNpArray=True)
print('loaded X_test with shape: {0}'.format(X_test.shape))
ids =  lib_IO.load_Y("/home/tg/Projects/LIS-Data/pr2/test.csv", usecols=[0], asNpArray=True)
print('loaded Ids with shape: {0}'.format(ids.shape))

for name, clf, param in zip(names, classifiers, param_grid):

        now = datetime.datetime.now()
        date_time = '{0:02d}_{1:02d}_{2:02d}_{3:02d}_{4:02d}'.format((now.year%2000),
                                                                     now.month, now.day,
                                                                     now.hour, now.minute)
        print('\ntesting classifier {0} start at {1}'.format(name,date_time))

        #normalize
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        #feature select


        #train all options
        if name is "Naive_Bayes" or name is "Linear_Discriminant_Analysis" or name is "Quadratic_Discriminant_Analysis":
            #if no gridsearch neccessary just do cross validation
            try:
                score = cross_val_score(clf, X_train, y_train, scoring='accuracy',cv=10,n_jobs=-1)
            except:
                print('error - Problem while doing cross validation with {0}'.format(name))
                traceback.print_exc(file=sys.stdout)
                score = 0
            finally:
                clf_tmp = clf

            print('-> {0} scored {1:.4f} (+/- {2:.4f})'.format(name,score.mean(),score.std()))
            lib_IO.log_score("/home/tg/Projects/LIS/Data/pr2/log.txt",date_time,name,score)
        else:
            #if gridsearch necessary do it and print best solution
            try:
                grid_search = GridSearchCV(clf, param, scoring='accuracy',cv=10,n_jobs=-1)
                grid_search.fit(X_train, y_train)
            except:
                print('error - Problem while doing gridsearch with {0}'.format(name))
                traceback.print_exc(file=sys.stdout)
                clf_tmp = clf
                score = 0
                best_param = 0
            else:
                clf_tmp = grid_search.best_estimator_
                score = grid_search.best_score_
                best_param = grid_search.best_params_

            print('-> {0} scored {1:.4f} with param: {2}'.format(name,score,best_param))
            lib_IO.log_best_param_score("/home/tg/Projects/LIS/Data/pr2/log.txt",date_time,name,score,best_param)

        print("-> finished CV starting with full training and prediction")

        try:
            clf_tmp.fit(X_train, y_train)
        except:
            print('error - Problem while training with full training set on {0}'.format(name))
            traceback.print_exc(file=sys.stdout)

        try:
            y_pred = clf_tmp.predict(X_test)
        except:
            print('error - Problem while predicting on test set with {0}'.format(name))
            traceback.print_exc(file=sys.stdout)
        else:
            lib_IO.write_Y('/home/tg/Projects/LIS/Data/pr2/handin-{0}-{1}.csv'.format(name,date_time),
                Y_pred=y_pred, Ids= ids)