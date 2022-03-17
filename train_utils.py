# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 21:13:56 2022

@author: benzener
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import tensorflow as tf
import complexnn
import os

def check_data_range(x):
    max_ = np.max(x)
    min_ = np.min(x)
    print(f'the largest value is {max_} and the smallest one is {min_}')
    
def show_fig(img, title_=None):
    plt.figure(dpi=300)
    plt.imshow(envelope_detection(img), cmap='gray')
    plt.title(title_)
    if title_ is not None:
        assert isinstance(title_, str)
        plt.title(title_)
    plt.show()

def envelope_detection(signal, DR=None):
    # check rank
    if len(signal.shape) == 4: 
        signal = signal.reshape(signal.shape[1:])
    assert len(signal.shape) == 3
    
    # check dtype
    if not np.isreal(signal).all():
        raise TypeError('signal must be an one-channel or two-channel real-valued array')
        
    # check last dimension
    if signal.shape[-1] == 2:
        envelope = np.sqrt(signal[:,:,0]**2 + signal[:,:,1]**2)
    elif signal.shape[-1] == 1 and np.isreal(signal).all():
        envelope = np.abs(scipy.signal.hilbert(signal))
    else:
        raise TypeError(f'Ivalid shape {signal.shape} for signal')
        
    if DR is None:
        return envelope
    else:
        dB_img = 20*np.log10(envelope/np.max(envelope)+1e-16) + DR
        return dB_img  

def set_env():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

def get_custom_object():
    custom_object = {
        'ComplexConv2D':complexnn.conv_test.ComplexConv2D,
        'ComplexBatchNormalization':complexnn.bn_test.ComplexBatchNormalization,
        'ComplexMSE':complexnn.loss.ComplexMSE
        }
    return custom_object