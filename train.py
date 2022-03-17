# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 21:24:37 2022

@author: benzener
"""

from train_utils import check_data_range, show_fig, envelope_detection, set_env, get_custom_object
from dataset import DataPreprocessing
from model import Model
import tensorflow as tf
import matplotlib.pyplot as plt

'''
    This file is main script running model and plotting outputs.
'''

# check and allow gpu memory to grow
set_env()
# call preprocessing function which returns traning data
preprocessing = DataPreprocessing()
(x_train, y_train), (x_test, y_test), (ideal_train, ideal_test) = preprocessing.get_data()
indices = preprocessing.indices
custom_object = get_custom_object

load_model         = False
filters            = 16
size               = 3
batch_size         = 2
lr                 = 1e-4
epochs             = 2
validation_split   = 0.3
activation         = 'cLeakyReLU'
loss               = 'cMSE'
forward            = False
nfig               = 3

if load_model:
    if forward:
        model = tf.keras.models.load_model('modelF_bnT_100.h5',custom_objects=custom_object)
        prediction = model.predict([x_test, ideal_test])
    else:
        model = tf.keras.models.load_model('model_bnT_200.h5',custom_objects=custom_object)
        prediction = model.predict(x_test)
else:
    UNet = Model(filters=filters,
                 size=size,
                 batch_size=batch_size,
                 lr=lr,
                 epochs=epochs,
                 validation_split=validation_split,
                 activations=activation,
                 losses=loss)
    if forward:
        model, history = UNet([x_train, ideal_train], y_train)
        prediction = model.predict([x_test, ideal_test])
    else:
        model, history = UNet(x_train, y_train)
        prediction = model.predict(x_test)
        
plt.figure()
plt.plot(history.history['loss'])
plt.title('loss')
plt.show()

nfig = 3
show_fig(x_test[nfig], 'speckle')
show_fig(prediction[nfig], 'predicted image')
show_fig(y_test[nfig], 'psf')
tf.keras.backend.clear_session()