# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 21:13:56 2022

@author: benzener
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as Signal
from scipy import io
import tensorflow as tf
import complexnn
import os
from datetime import datetime
import time

def check_data_range(x):
    max_ = np.max(x)
    min_ = np.min(x)
    print(f'the largest value is {max_} and the smallest one is {min_}')
    
def show_fig(img, ind, title_=None, DR=None):
    if len(img.shape) == 4:
        if img.shape[0] != 1:
            raise ValueError('cannot show over one image at once')
        img = img.reshape(img.shape[1:])
    assert len(img.shape) == 3
    
    axis = get_axis(img, ind)
    
    plt.figure(dpi=300)
    if DR:
        plt.imshow(envelope_detection(img, DR),
                   cmap='gray',
                   vmin=0,vmax=DR,
                   extent=axis,
                   aspect='auto')
        plt.colorbar()
    else:
        plt.imshow(envelope_detection(img), cmap='gray',extent=axis,aspect='auto')
    if title_ is not None:
        assert isinstance(title_, str)
        plt.title(title_)
    plt.xlabel('Lateral position (mm)')
    plt.ylabel('Depth (mm)')
    plt.ylim(axis[3],axis[2])
    plt.show()

def envelope_detection(signal, DR=None):
    
    # check dtype
    if not np.isreal(signal).all():
        raise TypeError('signal must be an one-channel or two-channel real-valued array')
        
    # check last dimension
    if signal.shape[-1] == 2:
        envelope = np.sqrt(signal[:,:,0]**2 + signal[:,:,1]**2)
    elif signal.shape[-1] == 1 and np.isreal(signal).all():
        envelope = np.abs(Signal.hilbert(signal, axis=0)).reshape(signal.shape[:2])
    else:
        raise ValueError(f'Ivalid shape {signal.shape} for signal')
        
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
        'ComplexMSE':complexnn.loss.ComplexMSE,
        'ctanh':complexnn.activation.ctanh
        }
    return custom_object

def get_default(complex_network=True):
    if complex_network:
        decimation = 2
        size = (3,6)
        loss = 'ComplexMSE'
    else:
        decimation = 1
        size = (6,6)
        loss = 'MSE'
    return decimation, size, loss

def save_model(model, history, name):
    now = datetime.now()
    day_month_year = now.strftime("%d%m%Y")
    sub_dir = name + day_month_year
    saved_dir = os.path.join(r'./modelinfo', sub_dir)
    model_name = name + '.h5'
    if not os.path.exists(saved_dir):
        try:
            os.mkdir(saved_dir)
        except FileNotFoundError:
            os.makedirs(saved_dir)
    model_saved_path = os.path.join(saved_dir, model_name)
    model.save(model_saved_path)
    history_name = os.path.join(saved_dir, name + 'history.txt')
    with open(history_name, 'w') as f:
        f.write(str(history))
    plt.figure()
    plt.plot(history['loss'], label='training loss')
    try:
        plt.plot(history['val_loss'], label='val loss')
    finally:
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig(os.path.join(saved_dir,name+'.png'))
        plt.show()

def get_axis(img, ind):
    DIR = r'./simulation_straight'
    r, c = img.shape[:2]
    if (ind+1)%4 == 0:
        level = 4
    else:
        level = (ind+1)%4
    file_name = 'Data_' + str((ind+1)//4 + 1) + '_delay_' + str(level) + '.mat'
    file_path = os.path.join(DIR, file_name)
    data = io.loadmat(file_path)
    dx = data.get('dx') * (513//img.shape[1])
    dz = data.get('dz') * (513//img.shape[0])
    depth = data.get('depth')
    x_axis = np.linspace(0,dx*c-dx,c) * 1e3 # [mm]
    z_axis = np.linspace(0,dz*c-dz,c) * 1e3 + depth * 1e3 # [mm]
    xmin, xmax = (np.min(x_axis), np.max(x_axis))
    zmin, zmax = (np.min(z_axis), np.max(z_axis))
    return (xmin, xmax, zmin, zmax)
    
def inference(model, testset):
    time_collection = []
    for i in range(10):
        _ = model.predict(testset[i:i+10])
    for i in range(10):
        s = time.time()
        _ = model.predict(testset)
        e = time.time()
        time_collection.append((s-e)*1000)
    return np.mean(time_collection)
    
            
    






    