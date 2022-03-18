# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 21:24:37 2022

@author: benzener
"""

from train_utils import check_data_range, show_fig, envelope_detection, set_env, get_custom_object, load_data
from dataset import DataPreprocessing
from model import Model
import tensorflow as tf
import matplotlib.pyplot as plt

'''
    This file is main script running model and plotting outputs.
'''


LOAD_MODEL         = True    # whether load pretrained model, True or False
FORWARD            = True    # whether use forward UNet or not, True or False
CALLBACK           = False   # whether add callbacks in model fit, True or False

TRAINING_NUM       = 1400    # number of training data, max:1600
FILTERS            = 16      # number of filters for the shallowest conv2D layer
SIZE               = 3       # size of each Conv2D kernel
BATCH_SIZE         = 4       # mini-batch size
LR                 = 1e-4    # learning rate of optimizer
EPOCHS             = 50      # training epochs
VALIDATION_SPLIT   = 0.2     # ratio of validation data referred to training data. ratio = # of val_data/ # of training_data
NFIG               = 3       # show the n-th fig of speckle, target, or prediction
DR                 = 40      # dynamic range in dB

ACTIVATION         = 'cLeakyReLU'   # Hidden-layer activation function, it could be 'modeReLU', 'cReLU', ... etc
LOSS               = 'cMSE'         # loss function,it could be 'cMSE', 'cMAE', 'cRMS'

# check and allow gpu memory to grow
set_env()
# call preprocessing function which returns traning data
if TRAINING_NUM != 1400:
    preprocessing = DataPreprocessing(training_num=TRAINING_NUM)
    (x_train, y_train), (x_test, y_test), (ideal_train, ideal_test) = preprocessing.get_data()
    # get shuffle indices
    indices = preprocessing.indices
else:
    (x_train, y_train), (x_test, y_test), (ideal_train, ideal_test) = load_data()




if LOAD_MODEL:
    # custom model needs to add custom function
    custom_object = get_custom_object()
    if FORWARD:
        model = tf.keras.models.load_model('modelF_bnT_100.h5',custom_objects=custom_object)
        prediction = model.predict([x_test, ideal_test])
    else:
        model = tf.keras.models.load_model('model_bnT_200.h5',custom_objects=custom_object)
        prediction = model.predict(x_test)
else:
    if CALLBACK:
        if VALIDATION_SPLIT:
            callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', mode='min')]
        else:
            callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, monitor='loss', mode='min')]
    else:
        callbacks = None
            
    UNet = Model(filters=FILTERS,
                 size=SIZE,
                 batch_size=BATCH_SIZE,
                 lr=LR,
                 epochs=EPOCHS,
                 validation_split=VALIDATION_SPLIT,
                 activations=ACTIVATION,
                 losses=LOSS,
                 callbacks=callbacks)
    if FORWARD:
        model, history = UNet([x_train, ideal_train], y_train)
        prediction = model.predict([x_test, ideal_test])
    else:
        model, history = UNet(x_train, y_train)
        prediction = model.predict(x_test)
    plt.figure()
    plt.plot(history.history['loss'], label='training loss')
    if VALIDATION_SPLIT:
        plt.plot(history.history['val_loss'], label='val loss')
        plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
      


show_fig(x_test[NFIG], 'speckle (linear)')
show_fig(prediction[NFIG], 'predicted image (linear)')
show_fig(y_test[NFIG], 'psf (linear)')

show_fig(x_test[NFIG], 'speckle (dB)', DR)
show_fig(prediction[NFIG], 'predicted image (dB)', DR)
show_fig(y_test[NFIG], 'psf (dB)', DR)


tf.keras.backend.clear_session()