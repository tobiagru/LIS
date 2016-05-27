#project 4 start
import numpy as np
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
import lib_IO
import datetime
import sys
import logging
import time
import h5py
from semisup.frameworks.CPLELearning import CPLELearningModel
from semisup.methods import scikitTSVM

logging.basicConfig(stream=sys.stdout,level=logging.DEBUG)
logging.info("start pr4")

path = "/home/tg/Projects/LIS/Data/pr4/"

train_labeled = h5py.File(path + "train_labeled.h5")
train_unlabeled = h5py.File(path + "train_unlabeled.h5")

X_train = np.zeros((28000,128), dtype=np.float16)
y_train = np.ones((28000,1), dtype=np.int8) * (-1)
X_valid = np.zeros((2000,128), dtype=np.float16)
y_valid = np.zeros((2000,1), dtype=np.int8)

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

train_labeled.close()
train_unlabeled.close()

#paramgrid
params = {
                 "gamma": np.logspace(-3,2,6),
                 "C": np.logspace(-3,2,6),
         },

name = "RBF_SVM",

score_max = 0.0
best_param = None
type = "..."

param_grid = list(ParameterGrid(params))
for param in param_grid:
    logging.info("start with gridsearch ")
    # model = scikitTSVM.SKTSVM(kernel='rbf',
    #                           gamma = param["gamma"],
    #                           lamU = param["C"],
    #                           )
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_valid)
    # score = accuracy_score(y_valid, y_pred, True)
    # logging.info("TSVM - gamma: {0:.5f}  C:{1:.5f}  -  score:{2:.4f}".format(param["gamma"], param["C"], score))
    # if score > score_max:
    #     score_max = score
    #     best_param = param
    #     type = "TSVM"
    # model = None
    # time.sleep(15)

    clf = SVC(kernel='rbf',
               probability=True,
               decision_function_shape='ovr',
               C=param["C"],
               gamma=param["gamma"])
    model = CPLELearningModel(clf, predict_from_probabilities=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    score = accuracy_score(y_valid, y_pred, True)
    logging.info("TSVM - gamma: {0:.5f}  C:{1:.5f}  -  score:{2:.4f}".format(param["gamma"], param["C"], score))
    if score > score_max:
        score_max = score
        best_param = param
        type = "pesse"
    clf = None
    model = None
    time.sleep(15)


    # clf = SVC(kernel='rbf',
    #                        probability=True,
    #                        decision_function_shape='ovr',
    #                        C=param["C"],
    #                        gamma=param["gamma"])
    # model = CPLELearningModel(clf, predict_from_probabilities=True, pessimistic = False)
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_valid)
    # score = accuracy_score(y_valid, y_pred, True)
    # logging.info("TSVM - gamma: {0:.5f}  C:{1:.5f}  -  score:{2:.4f}".format(param["gamma"], param["C"], score))
    # if score > score_max:
    #     score_max = score
    #     best_param = param
    #     type = "opti"
    # clf = None
    # model = None
    # time.sleep(15)

#--------------------------
X_train = None
y_train = None
X_valid = None
y_valid = None

X_train = np.zeros((30000,128), dtype=np.float16)
y_train = np.ones((30000,1), dtype=np.int8) * (-1)
X_test = np.zeros((8000,128), dtype=np.float16)
ids = np.zeros((8000,), dtype=np.uint16)

train_labeled = h5py.File(path + "train_labeled.h5")
train_unlabeled = h5py.File(path + "train_unlabeled.h5")
test = h5py.File(path + "test.h5")

train_labeled["train/block0_values"].read_direct(X_train,
                                                 dest_sel=np.s_[0:9000])
train_unlabeled["train/block0_values"].read_direct(X_train,
                                                 dest_sel=np.s_[9000:])
train_labeled["train/block1_values"].read_direct(y_train)
y_train = y_train.flatten()
y_valid = y_valid.flatten()
test["test/block0_values"].read_direct(X_test)
test["test/axis1"].read_direct(ids)

train_labeled.close()
train_unlabeled.close()
test.close()

if type == "TSVM":
    model = scikitTSVM.SKTSVM(kernel='rbf',
                              gamma = best_param["gamma"],
                              lamU = best_param["C"],
                              )
elif type == "pesse":
    clf = SVC(kernel='rbf',
                           probability=True,
                           decision_function_shape='ovr',
                           C=best_param["C"],
                           gamma=best_param["gamma"])
    model = CPLELearningModel(clf, predict_from_probabilities=True)
elif type == "opti":
    clf = SVC(kernel='rbf',
                           probability=True,
                           decision_function_shape='ovr',
                           C=best_param["C"],
                           gamma=best_param["gamma"])
    model = CPLELearningModel(clf, predict_from_probabilities=True, pessimistic = False)
else:
    model = None
    exit(-2)

model.fit(X_train, y_train)
y_test = model.predict(X_test)
y_test = y_test.astype(np.uint32)

now = datetime.datetime.now()
date_time = '{0:02d}_{1:02d}_{2:02d}_{3:02d}_{4:02d}'.format((now.year%2000),
                                                                 now.month, now.day,
                                                                 now.hour, now.minute)

lib_IO.write_Y("Data/pr4/{0}_{2}_{3}".format(name,best_param,date_time),y_test,Ids=ids)

