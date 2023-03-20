# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 15:52:53 2022

@author: benzener
"""


import os
if __name__ == '__main__':
    import sys
    currentpath = os.getcwd()
    addpath = os.path.dirname(os.path.dirname(currentpath))
    if addpath not in sys.path:
        sys.path.append(addpath)
    from baseband.utils import data_utils
    from baseband.setting import constant
    sys.path.remove(addpath)
else:
    from . import data_utils
    from ..setting import constant
import numpy as np
from scipy    import io
import complexnn 
import matplotlib.pyplot as plt
import tensorflow as tf

def get_custom_object():
    custom_object = {
        'ComplexConv2D':complexnn.conv_test.ComplexConv2D,
        'ComplexBatchNormalization':complexnn.bn_test.ComplexBatchNormalization,
        'ComplexMSE':complexnn.loss.ComplexMSE,
        'ctanh':complexnn.activation.ctanh,
        'FLeakyReLU': complexnn.activation.FLeakyReLU
        }
    return custom_object


def get_level(ind):
    '''
    Find their aberration level according to the index.
    ----------
    Arg:
        ind: int,the i-th simulation data.
    Return
        level 

    '''
    if not isinstance(ind, np.ndarray):
        ind = np.asarray(ind)
    ind = ind + 1
    levels = ind%constant.k
    try:
        # if input ind is a sequence
        levels[levels==0] = constant.k
        return levels
    except TypeError:
        # if input ind only an integer 
        if levels%constant.k == 0:
            return constant.k
        else:
            return levels%constant.k
    

    
def get_filename(ind):
    '''
    Find its file name according to the index.
    ----------
    Arg:
        ind: int, the i-th simulation data       
    Return:
        file_name: string, file name of the i-th simulation data
        
    '''
    if not isinstance(ind, np.ndarray):
        ind = np.asarray(ind)
    if np.size(ind) > 1:
        # if ind is a sequence
        file_names = []
        for ii in range(len(ind)):
            file_names.append(get_filename(ind[ii]))
        return file_names
    levels = get_level(ind)
    file_name = 'Data_' + str(ind//constant.k + 1) + '_delay_' + str(levels) + '.mat'
    return file_name

def get_data(ind, key):
    '''
    Find its data according to the index.
    ----------
    Arg:
        ind: int, the i-th simulation data
        key: string, any parameter in simulation data
    Return:
        file_name: string, file name of the i-th simulation data
        
    '''
    if not isinstance(key, str):
        raise TypeError(f"Key word must be a string type, but get {type(key)}")
    else:
        file_name = get_filename(ind)
        file_path = os.path.join(constant.DATAPATH, file_name)
        data = io.loadmat(file_path) # reading file gets information
        return data.get(key)

def get_axis(img, ind):
    '''
    Getting the physical axis or smapling rate. obtain the
    information about depth, dx, dz, or even sampling rate.
        Args: 
            img: A numpy array. 
            ind: An integer, the index of image.
            fs: Boolean.
        Returns:
            image axis if fs is false, otherwise the sampling rate.
    '''
    img = data_utils.precheck_dim(img)
    H, W = img.shape[1:-1]
    dx = get_data(ind, 'dx') * (constant.DATASIZE[2]/W)
    dz = get_data(ind, 'dz') * (constant.DATASIZE[1]/H)
    depth = get_data(ind, 'depth')
    x_axis = np.linspace(0,dx*W-dx,W) * 1e3 # [mm]
    z_axis = np.linspace(0,dz*H-dz,H) * 1e3 + depth * 1e3 # [mm]
    xmin, xmax = (np.min(x_axis), np.max(x_axis))
    zmin, zmax = (np.min(z_axis), np.max(z_axis))
    return (xmin, xmax, zmax, zmin)

def get_sampling_rate(img, ind):
    '''
    Find its sampling rate.
    ----------
    Arg:
        img: ndarray, the i-th simulation data
        ind: int, the i-th simulation data
    Return:
        floating scalar, sampling rate
        
    '''
    img = data_utils.precheck_dim(img)
    H = img.shape[1]
    dz = get_data(ind, 'dz') * (constant.DATASIZE[1]/H)
    return 1/(2*dz/constant.SOUNDV).reshape([-1])
    
def get_delaycurve(ind):
    '''
    Find its delay curve. Original delay curve is fixed in the range of [-0.5,0.5] pi. This function convert level k to
    its associated delay unit in pi. For example, k = 4 means its maximum phase error is 4/8 pi. So, we need to convert 
    [-0.5,0.5] to [-pi/2,pi/2].
    ----------
    Arg:
        ind: int, the i-th simulation data
    Return:
        ndarray, delay curve
    '''
    # obtain delay profile
    try:
        delay = get_data(ind, 'delay_curve')
        level = get_level(ind)
        if level != 1:
            return delay*level/8*2 # unit in pi
        else:
            return delay*0
    except TypeError:
        return 0
    
def get_soundv(ind, real=True):
    '''
    Find its true sound speed for filed II setting if real is True. 
    Otherwise this function returns sound speed for beamformation, 
    i.e. the wrong sound speed.
    
    '''
    try:
        if real:
            return np.squeeze([get_data(ii, 'soundv') for ii in range(len(ind))])
        else:
            return constant.SPEEDLIST[get_level(ind) - 1]
    except TypeError:
        return int(get_data(ind, 'soundv'))

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

def read_info(saved_name, saved_dir=None):
    '''
     reading the information saved by save_info function
     this function is in order to check whether the given
     augments are compatible with loaded model
    '''
    if saved_dir is not None:
        saved_path = os.path.join(saved_dir, saved_name)
    else:
        saved_path = saved_name
    if not os.path.exists(saved_path):
        raise FileNotFoundError('File does not exist')
    with open(saved_path, 'r') as f:
        content = f.read() # type of content is string
    return eval(content) # convert string to dict

def save_info(saved_var, saved_name, saved_dir=None):
    '''
    save the information of parameters, included 
    epoch. seed. and so on
    Args:
        model_name: saved path
        saved_var: parameters to be preserved
    '''
    # saved_dir = os.path.join(constant.MODELPATH, model_name)
    # file_name = model_name + '_parameters.txt'
    if saved_dir is not None:
        if not os.path.exists(saved_dir):
            try:
                os.mkdir(saved_dir)
            except FileNotFoundError:
                os.makedirs(saved_dir)
        saved_path = os.path.join(saved_dir, saved_name)
    else:
        saved_path = saved_name
    with open(saved_path, 'w') as f:
        f.write(str(saved_var))
        
def save_model(model, history, name):
    saved_dir = os.path.join(constant.MODELPATH, name)
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
    except KeyError:
        print('No validation')
    finally:
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig(os.path.join(saved_dir,name+'.png'))
        plt.show()

def progressbar(count, total, name):
    totalbar = 20
    ratio = count/total*100
    now = int((count/total)*totalbar)*'>'
    rest = (totalbar - int((count/total)*totalbar))*'.'
    print(f"\r{name} : [{now}{rest}] {ratio:.2f}% {count}/{total}", end='')
    if ratio == 100:
        print('\n')
    elif ratio >= 100:
        print('')
        
    


    


