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
    
def show_fig(img, title_=None, DR=None):
    plt.figure(dpi=300)
    if DR:
        plt.imshow(envelope_detection(img, DR), cmap='gray', vmin=0, vmax=DR)
        plt.colorbar()
    else:
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

def load_data():
    x_train = np.load('x_train.npy')
    y_train = np.load('y_train.npy')
    x_test = np.load('x_test.npy')
    y_test = np.load('y_test.npy')
    ideal_train = np.load('ideal_train.npy')
    ideal_test = np.load('ideal_test.npy')
    return (x_train, y_train), (x_test, y_test), (ideal_train, ideal_test)

def MSE(x, y):
    if len(x.shape) == 4:
        x = x.reshape(x.shape[1:])
    if len(y.shape) == 4:
        y = y.reshape(y.shape[1:])
    assert x.shape == y.shape
    err = x - y
    real = err[:,:,0]
    imag = err[:,:,1]
    return np.mean(real**2 + imag**2)

def SSIM(x, y, k1=0.01, k2=0.03, win_size=11, sigma=1.5, L=2):
    '''
    Parameters
    ----------
    x : np.float32
        input image.
    y : np.float32
        reference image.
    k1 : float, small value << 1
        a constant. The default is 0.01.
    k2 : float, small value << 1
        a constant. The default is 0.03.
    win_size : int, 
        Gaussian window kernel size. The default is 11.
    sigma : float,
        Gaussian window standard deviation. The default is 1.5.
    L : int, bitdepth
        range of input image, for normalized image, the largest value is 1,
        and the smallest value is -1. Hence L = 2. The default is 2.

    Returns
        SSIM

    '''
    if len(x.shape) == 4:
        x = x.reshape(x.shape[1:])
    if len(y.shape) == 4:
        y = y.reshape(y.shape[1:])
    assert x.shape == y.shape
    C1 = (k1*L)**2
    C2 = (K2*L)**2
    window = scipy.signal.windows.gaussian(win_size, std=sigma)
    