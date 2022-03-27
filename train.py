# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 21:24:37 2022

@author: benzener
"""

from train_utils import show_fig, set_env, get_custom_object, get_default, save_model
from newdataset import DataPreprocessing, GetData
from model import Model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
'''
    This file is main script running model and plotting outputs.
'''


LOAD_MODEL         = False    # whether load pretrained model, True or False
FORWARD            = False   # whether use forward UNet or not, True or False
CALLBACK           = False   # whether add callbacks in model fit, True or False
COMPLEX            = True  # whether use complex- or real-valued network, True or False
USING_DEFAULT      = False

NUM_TRAINING       = 1400    # number of training data, max:1600
DECIMATION         = 2       # downsample factor
FILTERS            = 8      # number of filters for the shallowest conv2D layer
SIZE               = (3,6)    # size of each Conv2D kernel
BATCH_SIZE         = 16       # mini-batch size
LR                 = 1e-4    # learning rate of optimizer
EPOCHS             = 200      # training epochs
VALIDATION_SPLIT   = 0.2     # ratio of validation data referred to training data. ratio = # of val_data/ # of training_data
NFIG               = 3       # show the n-th fig of speckle, target, or prediction
DR                 = 40      # dynamic range in dB

ACTIVATION         = 'LeakyReLU'   # Hidden-layer activation function, it could be 'modeReLU', 'cReLU', ... etc
LOSS               = 'ComplexMSE'  # loss function,it could be 'cMSE', 'cMAE', 'cRMS'

# check and allow gpu memory to grow
set_env()
if USING_DEFAULT:
    DECIMATION, SIZE, LOSS, BATCH_SIZE = get_default(COMPLEX)
# call preprocessing function which returns traning data
if NUM_TRAINING != 1400:
    preprocessing = DataPreprocessing()
    preprocessing.save_data()

get_dataset = GetData(factor=DECIMATION,
                      num_training=NUM_TRAINING,
                      complex_network=COMPLEX,
                      forward=FORWARD)
if FORWARD:
    (x_train, y_train), (x_test, y_test), (ideal_train, ideal_test) = get_dataset()
else:
    (x_train, y_train), (x_test, y_test) = get_dataset()




if LOAD_MODEL:
    # custom model needs to add custom function
    custom_object = get_custom_object()
    if FORWARD:
        model = tf.keras.models.load_model('modelF_bnT_50_valid.h5',custom_objects=custom_object)
        prediction = model.predict([x_test, ideal_test])
    else:
        model = tf.keras.models.load_model('model_bnT_200_valid.h5',custom_objects=custom_object)
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
                 callbacks=callbacks,
                 complex_network=COMPLEX)
    name = UNet.generate_name()
    if FORWARD:
        model, history = UNet([x_train, ideal_train], y_train)
        prediction = model.predict([x_test, ideal_test])
    else:
        model, history = UNet(x_train, y_train)
        prediction = model.predict(x_test)
    save_model(model, history.history, name)
      

level, ind =  get_dataset.find_level(NFIG, train=False)
show_fig(x_test[NFIG], ind, 'speckle (dB) ' + str(level), DR)
show_fig(prediction[NFIG], ind, 'predicted image (dB) ' + str(level), DR)
show_fig(y_test[NFIG], ind, 'psf (dB) ' + str(level), DR)

# metrics
mse = tf.keras.losses.MeanSquaredError()
print("MSE", mse(prediction[NFIG],y_test[NFIG]).numpy())
print("SSIM", tf.image.ssim(prediction[NFIG],y_test[NFIG],max_val=2).numpy())

tf.keras.backend.clear_session()