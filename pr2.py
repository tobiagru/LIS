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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import datetime
from sklearn.feature_selection import SelectFromModel
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier


names = [#"Nearest_Neighbors",
         #"Linear_SVM",
         "RBF_SVM",
         #"Poly_SVM",
         #"Sigmoid_SVM",
         #"Decision_Tree",
         #"Random_Forest",
         #"Extra_Trees",
         #"Gradient_Boosting"
         #"AdaBoost",
         #"Naive_Bayes",
         #"Linear_Discriminant_Analysis",
         #"Quadratic_Discriminant_Analysis"
        ]

classifiers = [
    #KNeighborsClassifier(),
    #SVC(kernel="linear"),
    SVC(kernel='rbf'),
    #SVC(kernel='poly'),
    #SVC(kernel='sigmoid'),
    #DecisionTreeClassifier(max_depth=5),
    #RandomForestClassifier(max_features=2),
    #ExtraTreesClassifier(),
    #GradientBoostingClassifier()
    #AdaBoostClassifier(),
    #GaussianNB(),
    #LinearDiscriminantAnalysis(),
    #QuadraticDiscriminantAnalysis()
    ]

param_grid = [#{'n_neighbors': np.arange(3,150,15,dtype=np.int16),'leaf_size':np.arange(5,20,dtype=np.int16)},
              #{'C': np.linspace(0.05,1,20)},
              {'estimator__C': np.logspace(-2,0.5,10),
               'estimator__gamma': np.logspace(-4,0.5,10),
               #'estimator__probability': [True,False],
               #'estimator__shrinking': [True,False],
               #'estimator__class_weight': ['balanced', None],
               },
              #{'C': np.linspace(0.05,1,5), 'gamma': np.linspace(0.05,5,5), 'degree': np.arange(2,4,dtype=np.int16)},
              #{'C': np.linspace(0.05,1,20), 'gamma': np.linspace(0.05,5,20)},
              #{'max_depth': np.arange(3,10,dtype=np.int16), 'max_features': np.arange(1,3,dtype=np.int16)},
              #{'max_depth': np.arange(8,15,dtype=np.int16), 'n_estimators': np.arange(300,400,10,dtype=np.int16)},
              # {'criterion': ['gini','entropy'],
              #  'max_features': [2,'auto','sqrt','log2'],
              #  'max_depth': np.arange(5,15,dtype=np.int16),
              #  'n_estimators': np.arange(150,700,50,dtype=np.int16),
              #  'min_samples_split': np.arange(1,3,dtype=np.int16),
              #  'min_samples_leaf':np.arange(1,5,dtype=np.int16),
              #  'min_weight_fraction_leaf': np.linspace(0.05,0.5,5),
              #  'bootstrap': [False,True],
              #  'oob_score': [False,True],
              #  'class_weight':['balanced','balanced_subsample',None]},
              # {'learning_rate': np.linspace(0.5,0.9,4),
              #  'max_features': [2,'auto','sqrt','log'],
              #  'n_estimators': np.arange(200,700,50,dtype=np.int32),
              #  'max_depth': np.arange(2,15,3,dtype=np.int16),
              #  'min_samples_split': np.arange(1,3,dtype=np.int16),
              #  'min_samples_leaf':np.arange(1,5,dtype=np.int16),
              #  'min_weight_fraction_leaf': np.linspace(0.05,0.5,5),
              #  'subsample': np.linspace(0.7,1.3,0.1)}
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
        #print("-> Normalize vectors")
        #scaler = StandardScaler().fit(X_train)
        #X_train = scaler.transform(X_train)
        #X_test = scaler.transform(X_test)

        mlt_clf = OneVsOneClassifier(clf)
        mlt_clf = OneVsRestClassifier(clf)

        #train all options
        print("-> do Gridsearch")
        if name is "Naive_Bayes" or name is "Linear_Discriminant_Analysis" or name is "Quadratic_Discriminant_Analysis":
            #if no gridsearch neccessary just do cross validation
            try:
                score = cross_val_score(clf, X_train, y_train, scoring='accuracy',cv=10,n_jobs=1)
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
                grid_search = GridSearchCV(mlt_clf, param, scoring='accuracy',cv=10, n_jobs=2)
                grid_search.fit(X_train, y_train)
            except:
                print('error - Problem while doing gridsearch with {0}'.format(name))
                traceback.print_exc(file=sys.stdout)
                clf_tmp = mlt_clf
                score = 0
                best_param = 0
            else:
                clf_tmp = grid_search.best_estimator_
                score = grid_search.best_score_
                best_param = grid_search.best_params_

            print('-> {0} scored {1:.4f} with param: {2}'.format(name,score,best_param))
            lib_IO.log_best_param_score("/home/tg/Projects/LIS/Data/pr2/log.txt",date_time,name,score,best_param)

        #feature select
        #print("-> feature Select from vectors")
        #threshold_y1 = "0.10*mean"
        #threshold_y2 = "0.06*mean"
        #trf1 = clf_tmp
        #trf1.fit(X_train,y_train)
        #try:
        #    selection = SelectFromModel(trf1, threshold= threshold_y1, prefit=True)
        #    X_train = selection.transform(X_train)
        #    X_test = selection.transform(X_test)
        #except:
        #    print('error - no selection method available with classifier')
        #    traceback.print_exc(file=sys.stdout)
        #print('-> number of features used after selection {0}'.format(np.shape(X_train)[1]))


        print("-> starting with full training and prediction")

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