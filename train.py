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
from train_utils import show_fig, envelope_detection, inference, phase_diff, projection
from train_utils import get_custom_object, get_default
from train_utils import save_model, save_metrics
from newdataset import DataPreprocessing, GetData
from model import Model
from DisplayedImg import VerifyPred, Fig, Difference, BasedCompute
import tensorflow as tf
import numpy as np

'''
    This file is main script running model and plotting outputs.
'''

LOAD_MODEL         = False   # whether load pretrained model, True or False
FORWARD            = False   # whether use forward UNet or not, True or False
CALLBACK           = False   # whether add callbacks in model fit, True or False
COMPLEX            = True  # whether use complex- or real-valued network, True or False
USING_DEFAULT      = False


NUM_TRAINING       = 1800    # number of training data, max:1600
DECIMATION         = 2      # downsample factor
FILTERS            = 16      # number of filters for the shallowest conv2D layer
SIZE               = (3,3)    # size of each Conv2D kernel
BATCH_SIZE         = 8       # mini-batch size
LR                 = 5*1e-4    # learning rate of optimizer
EPOCHS             = 200     # training epochs
VALIDATION_SPLIT   = 0     # ratio of validation data referred to training data. ratio = # of val_data/ # of training_data
NFIG               = 93       # show the n-th fig of speckle, target, or prediction
DR                 = 60      # dynamic range in dB

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
            model_name = 'complexmodel_Notforward_198_SSIM_FLeakyReLU_05052022'
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
        save_model(model, history.history, model_name)
        # save_model(model, history, model_name)
level, ind =  get_dataset.find_level(NFIG, train=False)
V = VerifyPred(prediction[NFIG],
               y_test[NFIG],
               ind,
               DR, 
               model_name=model_name,
               DIR_SAVED=DIR_SAVED,
               DIR_SIMULATION=DIR_SIMULATION)
Fig().envelope_fig(x_test[NFIG], DR, model_name, ind, 'speckle' + str(level))
V.pred_img
V.truth_img
V.show_lateral_projection
V.show_axial_projection
V.show_complex_diff()
V.show_phase_diff()
Fig().error_boxplot(prediction, y_test, get_dataset, model_name)
# show_fig(prediction[NFIG], ind, 'prediction' + str(level), DR, model_name)
# show_fig(y_test[NFIG], ind, 'psf' + str(level), DR, model_name)

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
# print('inference time', inference(model, x_test))

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

# phase_diff(y_test[NFIG], prediction[NFIG], 0.5, model_name)
save_metrics(envelope_pred, envelope_true, model_name)
# projection(prediction[NFIG], y_test[NFIG], model_name)
# projection(prediction[NFIG], y_test[NFIG], model_name, direction='axial')
tf.keras.backend.clear_session()