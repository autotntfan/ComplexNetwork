# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 21:24:37 2022

@author: benzener
"""

from train_utils import show_fig, save_model, envelope_detection, inference, save_metrics
from train_utils import set_env, get_custom_object, get_default
from newdataset import DataPreprocessing, GetData
from model import Model
import tensorflow as tf
import numpy as np
import os
'''
    This file is main script running model and plotting outputs.
'''


LOAD_MODEL         = True    # whether load pretrained model, True or False
FORWARD            = False   # whether use forward UNet or not, True or False
CALLBACK           = False   # whether add callbacks in model fit, True or False
COMPLEX            = False  # whether use complex- or real-valued network, True or False
USING_DEFAULT      = False

NUM_TRAINING       = 1400    # number of training data, max:1600
DECIMATION         = 1       # downsample factor
FILTERS            = 16      # number of filters for the shallowest conv2D layer
SIZE               = (6,3)    # size of each Conv2D kernel
BATCH_SIZE         = 8       # mini-batch size
LR                 = 1e-4    # learning rate of optimizer
EPOCHS             = 2      # training epochs
VALIDATION_SPLIT   = 0.2     # ratio of validation data referred to training data. ratio = # of val_data/ # of training_data
NFIG               = 93       # show the n-th fig of speckle, target, or prediction
DR                 = 60      # dynamic range in dB

ACTIVATION         = 'LeakyReLU'  # Hidden-layer activation function, it could be 'modeReLU', 'cReLU', ... etc
LOSS               = 'SSIM'  # loss function,it could be 'cMSE', 'cMAE', 'cRMS'
DIR_SAVED          = r'./modelinfo'
DIR_SIMULATION     = r'./simulation_straight'
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
        if COMPLEX:
            # model_name = 'complexmodel_Notforward_200_ComplexMSE_LeakyReLU_13042022'
            # model_name = 'complexmodel_Notforward_200_ComplexMSE_LeakyReLU_30032022'
            # model_name = 'complexmodel_Notforward_200_ComplexMSE_LeakyReLU_10042022'
            # model_name = 'complexmodel_Notforward_200_SSIM_MSE_LeakyReLU_20042022'
            # model_name = 'complexmodel_Notforward_200_SSIM_LeakyReLU_18042022'
            # model_name = 'complexmodel_Notforward_300_SSIM_LeakyReLU_22042022'
            model_name = 'complexmodel_Notforward_200_SSIM_LeakyReLU_22042022'
            # model_name = 'complexmodel_Notforward_200_MS_SSIM_LeakyReLU_23042022'
        else:
            # model_name = 'realmodel_Notforward_200_MSE_LeakyReLU_30032022'
            # model_name = 'realmodel_Notforward_200_SSIM_LeakyReLU_19042022'
            # model_name = 'realmodel_Notforward_100_SSIM_MSE_LeakyReLU_20042022'
            model_name = 'realmodel_Notforward_300_SSIM_LeakyReLU_26042022'
        model = tf.keras.models.load_model(os.path.join(DIR_SAVED,model_name,model_name+'.h5'),
                                           custom_objects=custom_object)
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
    model_name = UNet.generate_name()
    if FORWARD:
        model, history = UNet([x_train, ideal_train], y_train)
        prediction = model.predict([x_test, ideal_test])
    else:
        model, history = UNet(x_train, y_train)
        prediction = model.predict(x_test)
    save_model(model, history.history, model_name)
      

level, ind =  get_dataset.find_level(NFIG, train=False)
show_fig(x_test[NFIG], ind, 'speckle' + str(level), DR, model_name)
show_fig(prediction[NFIG], ind, 'prediction' + str(level), DR, model_name)
show_fig(y_test[NFIG], ind, 'psf' + str(level), DR, model_name)

# metrics
mse = tf.keras.losses.MeanSquaredError()

shape = (1,) + prediction[NFIG].shape[:-1] + (1,)
envelope_pred = envelope_detection(prediction[NFIG]).reshape(shape)
envelope_true = envelope_detection(y_test[NFIG]).reshape(shape)
    
print("MSE", mse(envelope_pred,envelope_true).numpy())
print("SSIM", tf.image.ssim(envelope_pred,
                            envelope_true,
                            max_val=1,
                            filter_size=7).numpy())
print("MS-SSIM", tf.image.ssim_multiscale(envelope_pred,
                                                  envelope_true,
                                                  max_val=1,
                                                  filter_size=7).numpy())
print('inference time', inference(model, x_test))

envelope_pred = np.expand_dims(envelope_detection(prediction),axis=-1)
envelope_true = np.expand_dims(envelope_detection(y_test),axis=-1)
print('average MSE',tf.reduce_mean(mse(envelope_pred,envelope_true)).numpy())
print("average SSIM", tf.reduce_mean(tf.image.ssim(envelope_pred,
                                                   envelope_true,
                                                   max_val=1,
                                                   filter_size=7)).numpy())
print("average MS-SSIM", tf.reduce_mean(tf.image.ssim_multiscale(envelope_pred,
                                                                 envelope_true,
                                                                 max_val=1,
                                                                 filter_size=7)).numpy())

save_metrics(envelope_pred, envelope_true, model_name)
tf.keras.backend.clear_session()