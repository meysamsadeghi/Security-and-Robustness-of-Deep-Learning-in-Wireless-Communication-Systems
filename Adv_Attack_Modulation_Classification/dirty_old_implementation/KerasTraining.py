# -*- coding: utf-8 -*-
"""
Modulation Classification
"""
import numpy as np
import sys
import pickle as cPickle
import tensorflow as tf
import keras
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt

# There is a Pickle incompatibility of numpy arrays between Python 2 and 3
# which generates ascii encoding error, to work around that we use the following instead of
# Xd = cPickle.load(open("RML2016.10a_dict.dat",'rb'))
with open('RML2016.10a_dict.dat','rb') as ff:
    u = cPickle._Unpickler( ff )
    u.encoding = 'latin1'
    Xd = u.load()

snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []  
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)

# Partition the data
#  into training and test sets of the form we can train/test on 
#  while keeping SNR and Mod labels handy for each
np.random.seed(2016)
n_examples = X.shape[0]
n_train = int(n_examples * 0.5)
train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test =  X[test_idx]



def to_onehot(yin):
    yy = list(yin) # This is a workaround as the map output for python3 is not a list
    yy1 = np.zeros([len(list(yy)), max(yy)+1])
    yy1[np.arange(len(list(yy))),yy] = 1
    return yy1
Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))




in_shp = list(X_train.shape[1:])
print(X_train.shape, in_shp)
classes = mods


#==============================================================================
# Build VT-CNN2 Neural Net model using Keras primitives -- 
#  - Reshape [N,2,128] to [N,1,2,128] on input
#  - Pass through 2 2DConv/ReLu layers
#  - Pass through 2 Dense layers (ReLu and Softmax)
#  - Perform categorical cross entropy optimization
dr = 0.5 # dropout rate (%)
model = models.Sequential()
model.add(Reshape([1]+in_shp, input_shape=in_shp))
model.add(ZeroPadding2D((0,2),data_format='channels_first'))
model.add(Convolution2D(256,(1,3), padding='valid', activation="relu", name="conv1",data_format='channels_first', kernel_initializer='glorot_uniform'))
model.add(Dropout(dr))
model.add(ZeroPadding2D((0,2),data_format='channels_first'))
model.add(Convolution2D(80,(2,3), padding="valid", activation="relu", name="conv2", data_format='channels_first', kernel_initializer='glorot_uniform'))
model.add(Dropout(dr))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1"))
model.add(Dropout(dr))
model.add(Dense(len(classes), kernel_initializer='he_normal', name="dense2" ))
model.add(Activation('softmax'))
model.add(Reshape([len(classes)]))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

#==============================================================================
# Set up some params 
nb_epoch = 100     # number of epochs to train on
batch_size = 1024  # training batch size

# perform training ...
#   - call the main training loop in keras for our network+dataset
filepath = 'convmodrecnets_CNN2_0.5.wts.h5'
history = model.fit(X_train,
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=2,
    validation_data=(X_test, Y_test),
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')]
    #callbacks = [
    #    keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
    #    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')]
    )
# we re-load the best weights once training is finished
#model.load_weights(filepath)













































