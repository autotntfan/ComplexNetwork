# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 21:24:37 2022

@author: benzener
"""
import tensorflow as tf
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
except RuntimeError as e:
    print(e)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from train_utils import get_custom_object, get_default
from train_utils import save_model, save_metrics
from newdataset import DataPreprocessing, GetData
from model import Model
from DisplayedImg import VerifyPred, Fig
from scipy import io
from computation import BasedCompute, Difference
from metrics import Metrics
import tensorflow as tf
import numpy as np


'''
    This file is main script running model and plotting outputs.
'''

LOAD_MODEL         = True  # whether load pretrained model, True or False
FORWARD            = False   # whether use forward UNet or not, True or False
CALLBACK           = False   # whether add callbacks in model fit, True or False
COMPLEX            = True  # whether use complex- or real-valued network, True or False
USING_DEFAULT      = False


NUM_TRAINING       = 1800    # number of training data, max:1600
DECIMATION         = 2      # downsample factor
FILTERS            = 16      # number of filters for the shallowest conv2D layer
SIZE               = (3,3)    # size of each Conv2D kernel
BATCH_SIZE         = 8       # mini-batch size
LR                 = 1e-4    # learning rate of optimizer
EPOCHS             = 200     # training epochs
VALIDATION_SPLIT   = 0.2     # ratio of validation data referred to training data. ratio = # of val_data/ # of training_data
NFIG               = 93       # show the n-th fig of speckle, target, or prediction
DR                 = 60      # dynamic range in dB
SEED               = 7414

ACTIVATION         = 'FLeakyReLU'  # Hidden-layer activation function, it could be 'modeReLU', 'cReLU', ... etc
LOSS               = 'SSIM'  # loss function,it could be 'cMSE', 'cMAE', 'cRMS'
DIR_SAVED          = r'./modelinfo'
DIR_SIMULATION     = r'./simulation_data'

if USING_DEFAULT:
    DECIMATION, SIZE, LOSS, BATCH_SIZE = get_default(COMPLEX)
# call preprocessing function which returns traning data
if NUM_TRAINING != 1800:
    preprocessing = DataPreprocessing()
    preprocessing.save_data()

get_dataset = GetData(factor=DECIMATION,
                      num_training=NUM_TRAINING,
                      complex_network=COMPLEX,
                      seed=SEED,
                      forward=FORWARD)
data_info = get_dataset.info()
if data_info['seed'] != SEED:
    raise ValueError('Dataset is not compatible')
B = BasedCompute()
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
            # model_name = 'complexmodel_Notforward_200_MS_SSIM_LeakyReLU_23042022'
            # model_name = 'complexmodel_Notforward_200_SSIM_MSE_LeakyReLU_29042022'
            # model_name = 'complexmodel_Notforward_200_SSIM_LeakyReLU_29042022'
            # model_name = 'complexmodel_Notforward_300_SSIM_LeakyReLU_30042022'
            # model_name = 'complexmodel_Notforward_300_SSIM_LeakyReLU_30042022_filter5'
            # model_name = 'complexmodel_Notforward_200_ComplexMSE_LeakyReLU_28042022'
            # model_name = 'complexmodel_Notforward_300_SSIM_LeakyReLU_01052022_filter15new'
            # model_name = 'complexmodel_Notforward_300_SSIM_LeakyReLU_01052022'
            # model_name = 'complexmodel_Notforward_300_SSIM_FLeakyReLU_04052022'
            # model_name = 'complexmodel_Notforward_200_SSIM_LeakyReLU_08052022'
            # model_name = 'complexmodel_Notforward_198_SSIM_FLeakyReLU_05052022'
            # model_name = 'complexmodel_Notforward_200_MS_SSIM_LeakyReLU_09052022'
            # model_name = 'complexmodel_Notforward_198_SSIM_FLeakyReLU_11052022'
            # model_name = 'complexmodel_Notforward_200_SSIM_FLeakyReLU_20052022'
            # model_name = 'complexmodel_Notforward_200_SSIM_FLeakyReLU_14052022'
            # model_name = 'complexmodel_Notforward_200_SSIM_LeakyReLU_21052022'
            # model_name = 'complexmodel_Notforward_200_ComplexMSE_LeakyReLU_22052022'
            # model_name = 'complexmodel_Notforward_200_ComplexMSE_FLeakyReLU_22052022'
            # model_name = 'complexmodel_Notforward_200_SSIM_FLeakyReLU_14052022'
            # model_name = 'complexmodel_Notforward_200_SSIM_MSE_FLeakyReLU_25052022'
            model_name = 'complexmodel_Notforward_200_SSIM_FLeakyReLU_30052022'
            # model_name = 'complexmodel_Notforward_200_SSIM_FLeakyReLU_13062022'
            # model_name = 'complexmodel_Notforward_198_SSIM_MSE_FLeakyReLU_24052022'
        else:
            # model_name = 'realmodel_Notforward_200_MSE_LeakyReLU_30032022'
            # model_name = 'realmodel_Notforward_200_SSIM_LeakyReLU_19042022'
            # model_name = 'realmodel_Notforward_100_SSIM_MSE_LeakyReLU_20042022'
            model_name = 'realmodel_Notforward_300_SSIM_LeakyReLU_26042022'
        # info = B.read_info(model_name)
        # if info['seed'] != SEED or info['validation_split'] != VALIDATION_SPLIT:
        #     raise ValueError('Dataset is not compatible')
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
                 validation_data=None,
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
        # model, history = UNet.running(x_train, y_train)
        prediction = model.predict(x_test)
    data_info.update(UNet.info())
    save_model(model, history.history, model_name)
        # save_model(model, history, model_name)
    B.save_info(model_name, data_info)
level, ind =  get_dataset.find_level(NFIG, train=False)
V = VerifyPred(prediction[NFIG],
               y_test[NFIG],
               ind,
               DR,
               model_name=model_name,
               DIR_SAVED=DIR_SAVED,
               DIR_SIMULATION=DIR_SIMULATION)
Fig().envelope_fig(x_test[NFIG], DR, 'speckle' ,ind, model_name, 'speckle' + str(level))
V.pred_img
V.truth_img
V.show_lateral_projection
V.show_axial_projection
V.show_complex_diff()
V.show_phase_diff()
Fig().err_fig(prediction, y_test, get_dataset, model_name)
Fig().topn_err_fig(prediction, y_test, get_dataset, 5, model_name)
# show_fig(prediction[NFIG], ind, 'prediction' + str(level), DR, model_name)
# show_fig(y_test[NFIG], ind, 'psf' + str(level), DR, model_name)

# metrics
M = Metrics()
M.save_info(prediction, y_test, model_name)
tf.keras.backend.clear_session()