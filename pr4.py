#project 4 start
import numpy as np
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import lib_IO
import datetime
import sys
import logging

logging.basicConfig(stream=sys.stdout,level=logging.DEBUG)
logging.info("start pr4")

#load stuff
train_labeled = pd.read_hdf("Data/pr4/train_labeled.h5", "train")
train_unlabeled = pd.read_hdf("Data/pr4/train_unlabeled.h5", "train")
test = pd.read_hdf("Data/pr4/test.h5", "test")

train, valid = train_test_split(train_labeled._values, test_size = 0.135, random_state = 17)

X_train = np.zeros(((21000 + train.shape[0]),128))
X_train[:train.shape[0]] = train[:, 1:129]
X_train[train.shape[0]:] = train_unlabeled._values[:, 0:128]

y_train = np.ones((21000 + train.shape[0],)) *(-1)
y_train[:train.shape[0]] = train[:, 0]

X_valid = valid[:,1:129]
y_valid = valid[:,0]
#k-means
#PCA
#bayes decision theory

#create classifier

#Pipeline

#paramgrid
params = [
             {
                 "kernel": ['rbf',],
                 "gamma": np.logspace(0.00001,100, 8),
                 "alpha": np.logspace(0.00001,100, 8),
                 "max_iter": [20,],
             },
             {
                 "kernel": ['knn',],
                 "n_neighbors": range(1,20,2),
                 "alpha": np.logspace(0.00001,100, 8),
                 "max_iter": [20,],
             },
             ]

param_grid = list(ParameterGrid(params))

names = ["propagation", "spreading"]

logging.info("start with training ")

for param in param_grid:
    for name in names:
        if param["kernel"] == 'rbf':
            if name == "propagation":
                clf = LabelPropagation(kernel=param["kernel"],
                                       gamma=["gamma"],
                                       alpha=param["alpha"],
                                       max_iter=param["max_iter"])
            else:
                clf = LabelSpreading(kernel=param["kernel"],
                                       gamma=["gamma"],
                                       alpha=param["alpha"],
                                       max_iter=param["max_iter"])
        else:
            if name == "propagation":
                clf = LabelPropagation(kernel=param["kernel"],
                                       n_neighbors=param["n_neighbors"],
                                       alpha=param["alpha"],
                                       max_iter=param["max_iter"])
            else:
                clf = LabelSpreading(kernel=param["kernel"],
                                       n_neighbors=param["n_neighbors"],
                                       alpha=param["alpha"],
                                       max_iter=param["max_iter"])

        now = datetime.datetime.now()
        date_time = '{0:02d}_{1:02d}_{2:02d}_{3:02d}_{4:02d}'.format((now.year%2000),
                                                                     now.month, now.day,
                                                                     now.hour, now.minute)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_valid)
        score = accuracy_score(y_valid, y_pred, False)

        #classification Type
        #ovo_clf = OneVsOneClassifier(clf)
        #ovr_clf = OneVsRestClassifier(clf)

        #Gridsearch
        #grid_search = GridSearchCV(clf, param, scoring='accuracy',cv=10, n_jobs=-1, verbose=1)
        #grid_search.fit(X_train, y_train)

        #clf_tmp = grid_search.best_estimator_
        #score = grid_search.best_score_
        #best_param = grid_search.best_params_

        lib_IO.log_best_param_score(datetime,name,score,param)