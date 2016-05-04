
import lib_IO
import numpy as np
import logging
import sys
import h5py
import datetime
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize

from keras import backend as K

from keras.models import Sequential

from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten, Reshape
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
#from keras.engine.training import batch_shuffle

from keras.optimizers import SGD


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

###########################
fname_train = "/home/tg/Projects/LIS/Data/pr3/train.h5"

fname_test = "/home/tg/Projects/LIS/Data/pr3/test.h5"

nb_epoch = 50

batch_size = 120

num_classes = 5
#num_classes = 1

classes = [0,1,2,3,4]

lr_rte = 0.01
nestrove = True

###############

logging.info("Loading dataset '{0}'".format(fname_train))
openfile = h5py.File(fname_train)

lab = openfile["train/block1_values"]
labels = np.zeros(lab.shape, dtype=np.uint8)
lab.read_direct(labels)
labels = label_binarize(labels, classes)

feat = openfile["train/block0_values"]
features = np.zeros(feat.shape, dtype=np.float32)
feat.read_direct(features)

i = openfile["train/axis1"]
ids = np.zeros(i.shape, dtype=np.uint32)
i.read_direct(ids)

logging.info("Closing dataset '{0}'".format(fname_train))
openfile.close()

logging.info("Loading dataset '{0}'".format(fname_test))
openfile = h5py.File(fname_test)

feat_test = openfile["test/block0_values"]
features_test = np.zeros(feat_test.shape, dtype=np.float32)
feat_test.read_direct(features_test)

i_test = openfile["test/axis1"]
ids_test = np.zeros(i_test.shape, dtype=np.uint32)
i_test.read_direct(ids_test)

logging.info("Closing dataset '{0}'".format(fname_test))
openfile.close()

features_train, features_valid,\
    labels_train, labels_valid,\
    ids_train, ids_valid = train_test_split(
    features, labels, ids,
    test_size=0.15 , random_state=17)

print("features_train shape: {0}".format(features_train.shape))
print("labels_train shape: {0}".format(labels_train.shape))
print("features_valid shape: {0}".format(features_valid.shape))
print("labels_valid shape: {0}".format(labels_valid.shape))
print("features_test shape: {0}".format(features_test.shape))

features = None
labels = None

###################################

def objective(self, y_true, y_pred):
        return K.categorical_crossentropy(y_pred, y_true)

optimizer = SGD(lr=lr_rte, nesterov=nestrove)


opts1 = [
    {
    "layer": "Embedding",
    "output_dim": 264,
    "input_dim": 100
    },
    {
    "layer": "LSTM",
    "output_dim": 128,
    "activation": "sigmoid",
    "inner_activation": "hard_sigmoid"
    },
    {
    "layer": "DropOut",
    #Dropout
    "p": 0.5,
    },
    {
    "layer": "Dense",
    #Dense
    "output_dim": num_classes,
    #Dense & Conv
    "activation": "sigmoid",
    },
]

opts2=[
    {
    "layer": "Dense",
    #Dense
    "output_dim": 1028,
    #Dense & Conv
    "activation": "linear",
    "input_dim": 100
    },
    {
    "layer": "Dense",
    #Dense
    "output_dim": 512,
    #Dense & Conv
    "activation": "relu",
    },
    {
    "layer": "DropOut",
    #Dropout
    "p": 0.4,
    },
    {
    "layer": "Dense",
    #Dense
    "output_dim": 265,
    #Dense & Conv
    "activation": "linear",
    },
    {
    "layer": "Dense",
    #Dense
    "output_dim": num_classes,
    #"output_dim": 1,
    #Dense & Conv
    "activation": "relu",
    },
    {
    "layer": "DropOut",
    #Dropout
    "p": 0.3,
    },
]

opts3 = [
    #recreate stacked LSTM sequence classification with stateful
]

opts4 = [
    {
    "layer": "Dense",
    #Dense
    "output_dim": 1028,
    #Dense & Conv
    "activation": "relu",
    "input_dim": 100
    },
    {
    "layer": "DropOut",
    #Dropout
    "p": 0.5,
    },
    {
    "layer": "Dense",
    #Dense
    "output_dim": 512,
    #Dense & Conv
    "activation": "relu",
    },
    {
    "layer": "DropOut",
    #Dropout
    "p": 0.5,
    },
    {
    "layer": "Dense",
    #Dense
    "output_dim": num_classes,
    #Dense & Conv
    "activation": "linear",
    },
]

opts5 = [
    {
    "layer": "Dense",
    #Dense
    "output_dim": 728,
    #Dense & Conv
    "activation": "tanh",
    "input_dim": 100
    },
    {
    "layer": "DropOut",
    #Dropout
    "p": 0.5,
    },
    {
    "layer": "Dense",
    #Dense
    "output_dim": 1028,
    #Dense & Conv
    "activation": "tanh",
    },
    {
    "layer": "DropOut",
    #Dropout
    "p": 0.5,
    },
    {
    "layer": "Dense",
    #Dense
    "output_dim": num_classes,
    #Dense & Conv
    "activation": "linear",
    },
]

mdl_cfgs = [

#    {"name": "LSTM_stacked_stateful", "opts": opts3},
    {"name": "MLP1", "opts": opts4},
#    {"name": "MLP2", "opts": opts5},
    {"name": "Multi_Dense", "opts": opts2},
    {"name": "LSTM_sequence", "opts": opts1},
]


###################################

for mdl_cfg in mdl_cfgs:

    mdl = Sequential()

    logging.info("initilized model {0}".format(mdl_cfg["name"]))

    start = True
    for opts in mdl_cfg["opts"]:

        if start is True:
            if opts["layer"] == "Dense":
                mdl.add(Dense(output_dim=opts["output_dim"],
                            init='normal',
                            input_dim=opts["input_dim"],
                            activation= opts["activation"],
                            W_regularizer=l2(0.001),
                            b_regularizer=l2(0.001)))

            elif opts["layer"] == "Conv":
                mdl.add(Convolution2D(nb_filter=opts["nb_filter"],
                                    nb_row=1,
                                    nb_col=opts["nb_col"],
                                    input_shape=[1,1,opts["input_dim"]],
                                    dim_ordering='th',
                                    subsample=[1,opts["subsample_length"]],
                                    activation= opts["activation"],
                                    init='normal',
                                    border_mode='valid',
                                    W_regularizer=l2(0.001),
                                    b_regularizer=l2(0.001)))
            elif opts["layer"] == "Embedding":
                mdl.add(Embedding(
                    output_dim=opts["output_dim"],
                    init='normal',
                    input_dim=opts["input_dim"],
                    W_regularizer=l2(0.001)
                ))
            elif opts["layer"] == "LSTM":
                mdl.add(LSTM(output_dim=opts["output_dim"],
                                init='normal',
                                input_dim=opts["input_dim"],
                                activation= opts["activation"],
                                inner_activation = opts["inner_activation"],
                                W_regularizer=l2(0.001),
                                b_regularizer=l2(0.001)))

            start = False

        else:
            if opts["layer"] == "Dense":
                mdl.add(Dense(output_dim=opts["output_dim"],
                                init='normal',
                                activation= opts["activation"],
                                W_regularizer=l2(0.001),
                                b_regularizer=l2(0.001)))

            elif opts["layer"] == "MaxPool":
                mdl.add(MaxPooling2D(pool_size=[1,opts["pool_length"]],
                                       dim_ordering='th',
                                       strides=None,
                                       border_mode='valid'))

            elif opts["layer"] == "DropOut":
                mdl.add(Dropout(p=opts["p"]))

            elif opts["layer"] == "Conv":
                mdl.add(Convolution2D(nb_filter=opts["nb_filter"],
                                        nb_row=1,
                                        nb_col=opts["nb_col"],
                                        subsample=[1,opts["subsample_length"]],
                                        dim_ordering='th',
                                        activation= opts["activation"],
                                        init='normal',
                                        border_mode='valid',
                                        W_regularizer=l2(0.001),
                                        b_regularizer=l2(0.001)))

            elif opts["layer"] == "Leaky":
                mdl.add(Activation(LeakyReLU(alpha=0.1)))
            elif opts["layer"] == "Flatten":
                mdl.add(Flatten())
            elif opts["layer"] == "LSTM":
                mdl.add(LSTM(output_dim=opts["output_dim"],
                                init='normal',
                                activation= opts["activation"],
                                inner_activation = opts["inner_activation"],
                                W_regularizer=l2(0.001),
                                b_regularizer=l2(0.001)))

        logging.info("Layer: " + opts["layer"] + " shape={0}".format(mdl.output_shape))




    mdl.add(Activation("softmax"))

    mdl.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["accuracy"])
    #mdl.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    logging.info("compiled model {0}".format(mdl_cfg["name"]))

    mdl.fit(x=features_train,
              y=labels_train,
              nb_epoch=nb_epoch,
              batch_size=batch_size,
              shuffle=True,
              verbose=1)
    logging.info("fit model {0}".format(mdl_cfg["name"]))

    #mdl.load_weights(filepath="/home/tg/Projects/LIS/weights_net2")
    #logging.info("load weights from {0}".format("/home/tg/Projects/LIS/weights_net2"))

    mdl.save_weights("/home/tg/Projects/LIS/Data/pr3/weights_" + mdl_cfg["name"] + "_{0}.h5".format(datetime.datetime.now()) , overwrite=False)
    logging.info("save weights as {0}".format("weights_" + mdl_cfg["name"] + "_{0}.h5".format(datetime.datetime.now())))

    score = mdl.evaluate(features_valid, labels_valid, batch_size= 32, verbose = 1)
    logging.info("test model {0} scored {1}".format(mdl_cfg["name"],score))

    labels_test = mdl.predict_classes(features_test, batch_size = 32, verbose = 1)
    # print(labels_test.shape)
    # print(labels_test[0:10])
    # labels_test_tmp = np.zeros([labels_test[0],],dtype=np.uint8)
    # for it in range(0,labels_test_tmp.shape[0]):
    #     for n in range(0,5):
    #         if labels_test[it,n] == 1:
    #             labels_test_tmp[it] = n
    # labels_test = labels_test_tmp
    # print(labels_test.shape)
    lib_IO.write_Y("/home/tg/Projects/LIS/Data/pr3/" + mdl_cfg["name"] + "_{0}.csv".format(datetime.datetime.now()), Y_pred=labels_test, Ids=ids_test)
    logging.info("/home/tg/Projects/LIS/Data/pr3/" + mdl_cfg["name"] + ".csv")