#project 4 start
import numpy as np
from sklearn.semi_supervised import LabelPropagation LabelSpreading
import pandas as pd

#load stuff
train_labeled = pd.read_hdf("train_labeled.h5", "train")
train_unlabeled = pd.read_hdf("train_unlabeled.h5", "train")
test = pd.read_hdf("test.h5", "test")

#k-means
#PCA
#bayes decision theory

#create classifier


#Pipeline


#paramgrid
param_grid = [
             {
                 "kernel": 'knn',
                 "n_neighbors": range(1,20),
                 "alpha": np.logspace(0.00001,100, 20),
                 "max_iter": [10,20,30,40],
             },
             {
                 "kernel": 'rbf',
                 "gamma": np.logspace(0.00001,100, 20),
                 "alpha": np.logspace(0.00001,100, 20),
                 "max_iter": [10,20,30,40],
             }
             ]


#classification Type
ovo_clf = OneVsOneClassifier(clf)
ovr_clf = OneVsRestClassifier(clf)

#Gridsearch
grid_search = GridSearchCV(mlt_clf, param, scoring='accuracy',cv=10, n_jobs=2)
grid_search.fit(X_train, y_train)
clf_tmp = grid_search.best_estimator_
score = grid_search.best_score_
best_param = grid_search.best_params_