#project 4 start
import numpy as np
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.grid_search import GridSearchCV
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

X_train = np.zeros((30000,128))
X_train[:9000] = train_labeled._values[:, 1:129]
X_train[9000:] = train_unlabeled._values[:, 0:128]

y_train = np.ones((30000)) *(-1)
y_train[:9000] = train_labeled._values[:, 0]

#k-means
#PCA
#bayes decision theory

#create classifier

#Pipeline

#paramgrid
param_grid = [
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

names = ["propagation", "spreading"]
clf1 = LabelPropagation()
clf2 = LabelSpreading()
clfs = [clf1, clf2]

for name, clf in zip(names,clfs):
    for param in param_grid:
        now = datetime.datetime.now()
        date_time = '{0:02d}_{1:02d}_{2:02d}_{3:02d}_{4:02d}'.format((now.year%2000),
                                                                     now.month, now.day,
                                                                     now.hour, now.minute)
        logging.info("start with {0}".format(name))
        #classification Type
        #ovo_clf = OneVsOneClassifier(clf)
        #ovr_clf = OneVsRestClassifier(clf)

        #Gridsearch
        grid_search = GridSearchCV(clf, param, scoring='accuracy',cv=10, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        clf_tmp = grid_search.best_estimator_
        score = grid_search.best_score_
        best_param = grid_search.best_params_
        lib_IO.log_best_param_score(datetime,name,score,best_param)