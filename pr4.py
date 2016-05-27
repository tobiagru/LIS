#project 4 start
import numpy as np
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import lib_IO
import datetime
import sys
import logging
import time
import h5py

logging.basicConfig(stream=sys.stdout,level=logging.DEBUG)
logging.info("start pr4")

# #load stuff
# train_labeled = pd.read_hdf("Data/pr4/train_labeled.h5", "train")
# train_unlabeled = pd.read_hdf("Data/pr4/train_unlabeled.h5", "train")
# test = pd.read_hdf("Data/pr4/test.h5", "test")
#
# train, valid = train_test_split(train_labeled._values, test_size = 0.135, random_state = 17)
#
# X_train = np.zeros(((21000 + train.shape[0]),128), dtype=np.float16)
#
# X_train[:train.shape[0]] = train[:, 1:129]
# X_train[train.shape[0]:] = train_unlabeled._values[:, 0:128]
#
# y_train = np.ones((21000 + train.shape[0],), dtype=np.float16) *(-1)
# y_train[:train.shape[0]] = train[:, 0]
#
# X_valid = valid[:,1:129]
# y_valid = valid[:,0]
#
# stdscl = StandardScaler()
# X_train = stdscl.fit_transform(X_train)
# X_valid = stdscl.transform(X_valid)
#
# X_test = test._values
# X_test = X_test.astype(np.float16
# ids = np.asarray(test.axes[0],dtype=np.uint32)
#
# train_labeled = None
# train_unlabeled = None
# test = None
path = "/home/tg/Projects/LIS/Data/pr4/"

train_labeled = h5py.File(path + "train_labeled.h5")
train_unlabeled = h5py.File(path + "train_unlabeled.h5")
test = h5py.File(path + "test.h5")

X_train = np.zeros((28000,128), dtype=np.float16)
y_train = np.ones((28000,1), dtype=np.int8) * (-1)
X_valid = np.zeros((2000,128), dtype=np.float16)
y_valid = np.zeros((2000,1), dtype=np.int8)
X_test = np.zeros((8000,128), dtype=np.float16)
ids = np.zeros((8000,), dtype=np.uint16)

train_labeled["train/block0_values"].read_direct(X_train,
                                                 source_sel=np.s_[0:7000],
                                                 dest_sel=np.s_[0:7000])
train_labeled["train/block0_values"].read_direct(X_valid,
                                                 source_sel=np.s_[7000:]
                                                 )
train_unlabeled["train/block0_values"].read_direct(X_train,
                                                 dest_sel=np.s_[7000:])
train_labeled["train/block1_values"].read_direct(y_train,
                                                 source_sel=np.s_[0:7000],
                                                 dest_sel=np.s_[0:7000])
train_labeled["train/block1_values"].read_direct(y_valid,
                                                 source_sel=np.s_[7000:])
y_train = y_train.flatten()
y_valid = y_valid.flatten()
test["test/block0_values"].read_direct(X_test)
test["test/axis1"].read_direct(ids)

train_labeled.close()
train_unlabeled.close()
test.close()

#paramgrid
params = [
             {
                 "kernel": ['rbf',],
                 "gamma": np.logspace(-3,2,6),
                 "alpha": [1, 0.8],
             },
             # {
             #     "kernel": ['knn',],
             #     "n_neighbors": [2,3,4,],
             #     "max_iter": [10,],
             # },
             ]



names = [
        #"propagation",
        "spreading",
        ]

for grid in params:
    param_grid = list(ParameterGrid(grid))
    for param in param_grid:
        for name in names:
            if param["kernel"] == 'rbf':
                if name == "propagation":
                    clf = LabelPropagation(kernel=param["kernel"],
                                           gamma=param["gamma"])
                else:
                    clf = LabelSpreading(kernel=param["kernel"],
                                           gamma=param["gamma"])
                extra_param = param["gamma"]
            else:
                if name == "propagation":
                    clf = LabelPropagation(kernel=param["kernel"],
                                           n_neighbors=param["n_neighbors"])
                else:
                    clf = LabelSpreading(kernel=param["kernel"],
                                           n_neighbors=param["n_neighbors"])
                extra_param = param["n_neighbors"]

            now = datetime.datetime.now()
            date_time = '{0:02d}_{1:02d}_{2:02d}_{3:02d}_{4:02d}'.format((now.year%2000),
                                                                         now.month, now.day,
                                                                         now.hour, now.minute)

            #classification Type
            #clf = OneVsOneClassifier(clf)
            #clf = OneVsRestClassifier(clf)

            logging.info("start with training ")
            clf.fit(X_train, y_train)
            #y_pred = clf.predict(X_valid)
            #print("min:{0}  max:{0}".format(y_pred.min(),y_pred.max()))
            #score = accuracy_score(y_valid, y_pred, True)

            print("found classes are {0}".format(clf.classes_))

            y_test = clf.predict(X_test)
            y_test = y_test.astype(np.uint32)

            lib_IO.write_Y("Data/pr4/{0}_{1}_{2}_{3}".format(name,param["kernel"],extra_param,date_time),y_test,Ids=ids)
            #Gridsearch
            #grid_search = GridSearchCV(clf, param, scoring='accuracy',cv=10, n_jobs=-1, verbose=1)
            #grid_search.fit(X_train, y_train)

            #clf_tmp = grid_search.best_estimator_
            #score = grid_search.best_score_
            #best_param = grid_search.best_params_

            #lib_IO.log_best_param_score(date_time,name,score,param)
            clf = None
            #time.sleep(30)