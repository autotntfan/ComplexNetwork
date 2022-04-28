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

DIR_SAVED          = r'./modelinfo'
DIR_SIMULATION     = r'./simulation_straight'

def check_data_range(x):
    max_ = np.max(x)
    min_ = np.min(x)
    print(f'the largest value is {max_} and the smallest one is {min_}')
    
def show_fig(img, ind=None, title_=None, DR=None, name=None):
    # check image dimension is 3 or 4
    if len(img.shape) == 4:
        if img.shape[0] != 1:
            raise ValueError('unable to support multiple images')
        shape = img.shape[1:-1]
    elif len(img.shape) == 3:
        shape = img.shape[:-1]
    else:
        raise ValueError('invalid image shape')
    if ind is None:
        axis = None
    else:
        axis = _get_axis(img, ind)
    plt.figure(dpi=300)
    envelope_img = envelope_detection(img, DR)
    envelope_img = envelope_img.reshape(shape)
    assert len(envelope_img.shape) == 2
    if DR is not None:
        plt.imshow(envelope_img,
                    cmap='gray',
                    vmin=0,vmax=DR,
                    extent=axis,
                    aspect='auto')
        plt.colorbar()
    else:
        plt.imshow(envelope_img, cmap='gray',extent=axis,aspect='auto')
    if title_ is not None:
        assert isinstance(title_, str)
        plt.title(title_)
    plt.xlabel('Lateral position (mm)')
    plt.ylabel('Depth (mm)')
    if name is not None:
        saved_name = os.path.join(DIR_SAVED, name , title_ + str(ind) + '.png')
        plt.savefig(saved_name, dpi=300)
    plt.show()

def envelope_detection(signal, DR=None):
    '''
    Parameters
    ----------
    signal : float32, numpy array
    DR : int, dynamic range. The default is None.
    Returns
    -------
    envelop-detected signal in float32. Its shape is same as input shape.
    '''
    # check dtype
    if not np.isreal(signal).all():
        raise ValueError('signal must be an one-channel or two-channel real-valued array')
    # check rank
    if len(signal.shape) == 3:
        signal = np.expand_dims(signal, axis=0)
    shape = signal.shape
    assert len(shape) == 4
    
    if shape[-1] == 2:
        envelope = np.sqrt(signal[:,:,:,0]**2 + signal[:,:,:,1]**2)
    elif shape[-1] == 1:
        envelope = np.abs(Signal.hilbert(signal, axis=1))
    else:
        raise ValueError('signal channels must be one or two')
    envelope = envelope.reshape(shape[:-1])
    ratio = np.max(envelope, axis=(1,2),keepdims=True)
    envelope = envelope/ratio
    if DR is None:
        return envelope
    else:
        dB_img = 20*np.log10(envelope + 1e-16) + DR
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
    batch = 8
    if complex_network:
        decimation = 2
        size = (3,6)
        loss = 'ComplexMSE'
    else:
        decimation = 1
        size = (6,6)
        loss = 'MSE'
    return decimation, size, loss, batch

def save_model(model, history, name):
    saved_dir = os.path.join(DIR_SAVED, name)
    model_name = name + '.h5'      # model
    model_figname = name + 'arc.png'  # model architecture 
    if not os.path.exists(saved_dir):
        try:
            os.mkdir(saved_dir)
        except FileNotFoundError:
            os.makedirs(saved_dir)
    model_saved_path = os.path.join(saved_dir, model_name)
    model_figpath = os.path.join(saved_dir, model_figname)
    # plot and save model architecture
    tf.keras.utils.plot_model(model, to_file=model_figpath, show_shapes=True, show_layer_names=True, dpi=900)
    model.save(model_saved_path)
    history_name = os.path.join(saved_dir, name + 'history.txt') # loss value per epoch
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
def save_metrics(envelope_pred, envelope_true, name):
    save_dir = os.path.join(DIR_SAVED, name)
    file_name = os.path.join(save_dir, name + '_metrics.txt')
    mse = tf.losses.MeanSquaredError()
    metrics = {
        'mse':tf.reduce_mean(mse(envelope_pred,envelope_true)).numpy(),
        'ssim':tf.reduce_mean(tf.image.ssim(envelope_pred,
                                            envelope_true,
                                            max_val=1,
                                            filter_size=7)).numpy(),
        'ms_ssim':tf.reduce_mean(tf.image.ssim_multiscale(envelope_pred,
                                                          envelope_true,
                                                          max_val=1,
                                                          filter_size=7)).numpy()
        }
    print(file_name)
    with open(file_name,'w') as f:
        f.write(str(metrics))
     
def _get_axis(img, ind, fs=False):
    assert len(img.shape) ==3
    r, c = img.shape[:2]
    if (ind+1)%4 == 0:
        level = 4
    else:
        level = (ind+1)%4
    file_name = 'Data_' + str((ind+1)//4 + 1) + '_delay_' + str(level) + '.mat'
    print(file_name)
    file_path = os.path.join(DIR_SIMULATION, file_name)
    data = io.loadmat(file_path)
    dx = data.get('dx') * (513/img.shape[1])
    dz = data.get('dz') * (513/img.shape[0])
    depth = data.get('depth')/2
    x_axis = np.linspace(0,dx*c-dx,c) * 1e3 # [mm]
    z_axis = np.linspace(0,dz*r-dz,r) * 1e3 + depth * 1e3 # [mm]
    xmin, xmax = (np.min(x_axis), np.max(x_axis))
    zmin, zmax = (np.min(z_axis), np.max(z_axis))
    if fs:
        return 1/(2*dz/1540).reshape([-1])
    else:
        return (xmin, xmax, zmax, zmin)
    
def inference(model, testset):
    time_collection = []
    for i in range(10):
        _ = model.predict(testset[i:i+10])
    for i in range(10):
        s = time.time()
        _ = model.predict(testset)
        e = time.time()
        time_collection.append(e-s)
    return np.mean(time_collection)/testset.shape[0]

def lateral_projection(signal):
    if len(signal.shape) == 4:
        projection = np.max(signal,axis=1)
    else:
        projection = np.max(signal,axis=0)
    plt.figure()
    plt.plot(projection)
    plt.show()
    return projection

def show_fft(signal, ind, Aline=False):
    # signal shape = [H,W,C]
    fs = _get_axis(signal, ind, fs=True)
    # signal shape = [H,W]
    signal = signal.reshape(signal.shape[:-1])
    if Aline:
        center = signal.shape[1]//2
        Signal = np.abs(np.fft.fftshift(np.fft.fft(signal[:,center], axis=0)))
    else:
        Signal = np.abs(np.fft.fftshift(np.fft.fft(signal, axis=0)))
    freq_axis = np.linspace(-fs/2,fs/2,Signal.shape[0])/1e6
    plt.figure()
    plt.plot(freq_axis,Signal)
    plt.xlabel('MHz')
    plt.show()

# class PostProcessing():
    
#     def __init__(self,
#                  model,
#                  name=None,
#                  DR=None,
#                  DIR_SAVED=r'./modelinfo',
#                  DIR_SIMULATION=r'./simulation_straight'):
#         self.model = model
#         self.name = name
#         self.DR = DR
        
#         self.DIR_SAVED = DIR_SAVED
#         self.DIR_SIMULATION = DIR_SIMULATION
        
#     def show_fig(self, img, ind=None, title_=None, saved=False):
        
#         # check image dimension is 3 or 4
#         if len(img.shape) == 4:
#             if img.shape[0] != 1:
#                 raise ValueError('unable to support multiple images')
#             shape = img.shape[1:-1]
#         elif len(img.shape) == 3:
#             shape = img.shape[:-1]
#         else:
#             raise ValueError('invalid image shape')
#         if ind is None:
#             axis = None
#         else:
#             axis = self._get_axis(img, ind)
#         plt.figure(dpi=300)
#         envelope_img = self.envelope_detection(img, self.DR)
#         envelope_img = envelope_img.reshape(shape)
#         assert len(envelope_img.shape) == 2
#         if self.DR is not None:
#             plt.imshow(envelope_img,
#                        cmap='gray',
#                        vmin=0,vmax=self.DR,
#                        extent=axis,
#                        aspect='auto')
#             plt.colorbar()
#         else:
#             plt.imshow(envelope_img, cmap='gray',extent=axis,aspect='auto')
#         if title_ is not None:
#             assert isinstance(title_, str)
#             plt.title(title_)
#         plt.xlabel('Lateral position (mm)')
#         plt.ylabel('Depth (mm)')
#         if saved:
#             saved_name = os.path.join(self.DIR_SAVED, self.name , title_ + '.png')
#             plt.savefig(saved_name, dpi=300)
#         plt.show()
      
#     def envelope_detection(self, signal):
        
#         '''
#         Parameters
#         ----------
#         signal : float32, numpy array
#         DR : int, dynamic range. The default is None.
#         Returns
#         -------
#         envelop-detected signal in float32. Its shape is same as input shape.
#         '''
#         # check dtype
#         if not np.isreal(signal).all():
#             raise ValueError('signal must be an one-channel or two-channel real-valued array')
#         # check rank
#         if len(signal.shape) == 3:
#             signal = np.expand_dims(signal, axis=0)
#         shape = signal.shape
#         assert len(shape) == 4
        
#         if shape[-1] == 2:
#             envelope = np.sqrt(signal[:,:,:,0]**2 + signal[:,:,:,1]**2)
#         elif shape[-1] == 1:
#             envelope = np.abs(Signal.hilbert(signal, axis=1))
#         else:
#             raise ValueError('signal channels must be one or two')
#         envelope = envelope.reshape(shape[:-1])
#         ratio = np.max(envelope, axis=(1,2),keepdims=True)
#         envelope = envelope/ratio
#         if self.DR is None:
#             return envelope
#         else:
#             dB_img = 20*np.log10(envelope + 1e-16) + self.DR
#             return dB_img
          
#     def save_model(self, history):
        
#         saved_dir = os.path.join(self.DIR_SAVED, self.name)
#         model_name = self.name + '.h5'      # model
#         model_arcname = self.name + 'arc.png'  # model architecture
#         if not os.path.exists(saved_dir):
#             try:
#                 os.mkdir(saved_dir)
#             except FileNotFoundError:
#                 os.makedirs(saved_dir)
#         model_saved_path = os.path.join(saved_dir, model_name)
#         model_arcpath = os.path.join(saved_dir, model_arcname)
#         # plot and save model architecture
#         tf.keras.utils.plot_model(self.model, to_file=model_arcpath, show_shapes=True, show_layer_names=True, dpi=900)
#         self.model.save(model_saved_path)
#         history_name = os.path.join(saved_dir, self.name + 'history.txt') # loss value per epoch
#         with open(history_name, 'w') as f:
#             f.write(str(history))
#         plt.figure()
#         plt.plot(history['loss'], label='training loss')
#         try:
#             plt.plot(history['val_loss'], label='val loss')
#         finally:
#             plt.legend()
#             plt.xlabel('epochs')
#             plt.ylabel('loss')
#             plt.savefig(os.path.join(saved_dir,self.name+'.png'))
#             plt.show()
            
#     def inference(self, testset):
        
#         time_collection = []
#         for i in range(10):
#             _ = self.model.predict(testset[i:i+10])
#         for i in range(10):
#             s = time.time()
#             _ = self.model.predict(testset)
#             e = time.time()
#             time_collection.append(e-s)
#         return np.mean(time_collection)/testset.shape[0]
    
#     def _get_axis(self, img, ind):
        
#         assert len(img.shape) ==3
#         r, c = img.shape[:2]
#         if (ind+1)%4 == 0:
#             level = 4
#         else:
#             level = (ind+1)%4
#         file_name = 'Data_' + str((ind+1)//4 + 1) + '_delay_' + str(level) + '.mat'
#         file_path = os.path.join(self.DIR_SIMULATION, file_name)
#         data = io.loadmat(file_path)
#         dx = data.get('dx') * (513/img.shape[1])
#         dz = data.get('dz') * (513/img.shape[0])
#         depth = data.get('depth')/2
#         x_axis = np.linspace(0,dx*c-dx,c) * 1e3 # [mm]
#         z_axis = np.linspace(0,dz*r-dz,r) * 1e3 + depth * 1e3 # [mm]
#         xmin, xmax = (np.min(x_axis), np.max(x_axis))
#         zmin, zmax = (np.min(z_axis), np.max(z_axis))
#         return (xmin, xmax, zmax, zmin)
    

      
    
        
    
            
    






    