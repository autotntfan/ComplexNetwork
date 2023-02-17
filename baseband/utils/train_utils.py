# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 21:13:56 2022

@author: benzener
"""
import complexnn
import os
import time
import numpy             as np
import tensorflow        as tf
import matplotlib.pyplot as plt
if __name__ == '__main__':
    import sys
    import pathlib
    currentpath = os.getcwd()
    path = pathlib.Path(currentpath).parts
    addpath = path[:-2]
    addpath = os.path.join(*addpath)
    if addpath not in sys.path:
        sys.path.append(addpath)
    from baseband.setting import constant
    sys.path.remove(addpath)
else:
    from ..setting import constant

def get_custom_object():
    '''
    When loading model, you need to specify the custom-designed layers in the
    model. Otherwise, Tensorflow will print out an error that can't find these
    objects.
    '''
    custom_object = {
        'ComplexConv2D':complexnn.conv_test.ComplexConv2D,
        'ComplexBatchNormalization':complexnn.bn_test.ComplexBatchNormalization,
        'ComplexMSE':complexnn.loss.ComplexMSE,
        'ctanh':complexnn.activation.ctanh,
        'FLeakyReLU': complexnn.activation.FLeakyReLU
        }
    return custom_object

def read_info(model_name):
    '''
     Reading the information saved by save_info function.
     This function is used to check whether the given
     augments are compatible with the loaded model, i.e. model_name.
    '''
    saved_dir = os.path.join(constant.DIR_SAVEMODEL, model_name)
    file_name = model_name + '_parameters.txt'
    if not os.path.exists(saved_dir):
        raise FileNotFoundError('file does not exist')
    saved_path = os.path.join(saved_dir, file_name)
    with open(saved_path, 'r') as f:
        content = f.read() # type of content is string
    return eval(content) # convert string to dict

def save_info(model_name, saved_var):
    '''
    save the information of parameters, included 
    epoch. seed. and so on
    Args:
        model_name: saved path
        saved_var: parameters to be preserved
    '''
    saved_dir = os.path.join(constant.DIR_SAVEMODEL, model_name) # constant.DIR_SAVEMODEL/model_name/..
    file_name = model_name + '_parameters.txt' 
    if not os.path.exists(saved_dir):
        try:
            os.mkdir(saved_dir)
        except FileNotFoundError:
            os.makedirs(saved_dir)
    saved_path = os.path.join(saved_dir, file_name) # constant.DIR_SAVEMODEL/model_name/file_name
    with open(saved_path, 'w') as f:
        f.write(str(saved_var))
    
def save_model(model, history, name):
    '''
    Save model information included model.h5, model architecture picture, 
    loss curve figure and history (.txt).
    
    Args:
        model: tensorflow model
        history: training loss
        name: model name consists of date and some parameter information
    '''
    saved_dir = os.path.join(constant.DIR_SAVEMODEL, name)
    if not os.path.exists(saved_dir):
        try:
            os.mkdir(saved_dir)
        except FileNotFoundError:
            os.makedirs(saved_dir)
    model_saved_path = os.path.join(saved_dir, name + '.h5')
    model_arcfigpath = os.path.join(saved_dir, name + 'arc.png')
    # plot and save model architecture
    tf.keras.utils.plot_model(model, to_file=model_arcfigpath, show_shapes=True, show_layer_names=True, dpi=900)
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

    



    

    