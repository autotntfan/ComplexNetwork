# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 01:31:10 2021

@author: benzener
"""

import complexnn
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Layer
import keras
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)
        

class AmplitudeMaxout_piece(Layer):
    def __init__(self, num_pieces, name='AMU',**kwargs):
        super().__init__(**kwargs)
        self.__name__ = name
        self.num_pieces = num_pieces    
  
    @tf.function
    def call(self, x, axis=None):
        shape = x.get_shape().as_list()   
        tf.print('input shape',tf.keras.backend.shape(x))
        if axis is None:
            axis = -1
            shape[0] = -1
        # nb of Re/Im channels
        num_channels = shape[axis]//2
        # if # of Re != Im channels
        if shape[axis]%2:
            raise ValueError(f'nb of real/imaginary channel are inequivalent')
        # Re/Im channel 每num_piece比較一次，因此最後會有num_units個channel    
        assert num_channels % self.num_pieces == 0
        self.num_units = num_channels//self.num_pieces
        # 增加一個dim，最後一個dim為比較大小用，最後會被壓縮，壓縮至倒數第二個channel，因此最後會有num_units個channel
        shape[axis] = self.num_units
        exp_shape = shape + [self.num_pieces]
        # 若輸入的Re/Im channel數量無法被num_pieces整除，需分開處理整除以及剩下的部分
        self.num_rest_piece = num_channels%self.num_pieces
#         if self.num_rest_piece:
#             # 將整除的部分取出
#             real_part = tf.reshape(x[:,:,:,:num_channels-self.num_rest_piece],exp_shape)
#             imag_part = tf.reshape(x[:,:,:,num_channels:-self.num_rest_piece],exp_shape)
#             # 將整除的部分找出AMU
#             real_part, imag_part = self.return_AMU(real_part, imag_part, exp_shape)
#             # -----------------------------------------------------------
#             # 處理剩下的部分，倒數第二個dim必定是1，最後一個為剩下的piece數量
#             rest_exp_shape = shape[0:3] + [1,self.num_rest_piece]
#             # 將剩餘的部分取出
#             rest_real_part = tf.reshape(x[:,:,:,num_channels-self.num_rest_piece:num_channels], rest_exp_shape)
#             rest_imag_part = tf.reshape(x[:,:,:,-self.num_rest_piece:], rest_exp_shape)
#             rest_real_part, rest_imag_part = self.return_AMU(rest_real_part, rest_imag_part, rest_exp_shape)
#             real = tf.concat([real_part,rest_real_part], axis=-1)
#             imag = tf.concat([imag_part,rest_imag_part], axis=-1)
            
#             return tf.concat([real,imag],axis=-1)         
#         else:
            #tf.print(exp_shape)
        tf.print('expand shape',exp_shape)
        real_part = tf.reshape(x[:,:,:,:num_channels],exp_shape)
        imag_part = tf.reshape(x[:,:,:,num_channels:],exp_shape)
        real_part, imag_part = self.return_AMU(real_part, imag_part, exp_shape)
        tf.print('concat shape',tf.shape(tf.concat([real_part,imag_part],axis=-1)))
        return tf.concat([real_part,imag_part],axis=-1)

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        if self.num_rest_piece:
            shape[-1] = 2*self.num_units + 1
        else:
            shape[-1] = 2*self.num_units
        return tuple(shape)
    
    def get_config(self):
        base_config = super().get_config()
        config = {'units': self.num_units,
                  'pieces': self.num_pieces}
        return dict(list(base_config.items()) + list(config.items()))
    
    def return_AMU(self, real_part, imag_part ,expand_shape):

        # 計算複數絕對值長度
        modulus = real_part**2 + imag_part**2
        # 將最後兩個dim重組成(num_units,num_pieces)
        expand_modulus = tf.reshape(modulus, expand_shape)
        # 找出最大值後重組成原本的dim再根據位置判斷哪個實部虛部造成該最大值
        cond = tf.where(expand_modulus==tf.expand_dims(tf.reduce_max(expand_modulus, axis=-1),-1),True,False)
        tf.print('expand:',tf.shape(tf.expand_dims(tf.reduce_max(expand_modulus, axis=-1),-1)))
        tf.print('cond shape',tf.shape(cond))
        # 過濾出實部虛部，須注意使用[]過濾出的值會是1D，必須再重組
        real_part = tf.reshape(real_part[cond],expand_shape[:-1])
        imag_part = tf.reshape(imag_part[cond],expand_shape[:-1])
        
        return real_part, imag_part

def ComplexRMS(y_true, y_pred):       
    if y_true.get_shape() != y_pred.get_shape():
        if y_true.get_shape()[1] is None:
            pass
        else:
            raise ValueError(f"label has {y_true.get_shape()} but input has {y_predict.get_shape()}")
    shape = y_pred.get_shape().as_list()
    if shape[-1]%2:
        raise ValueError(f"nb of imaginary part isn't equal to that of real part")
    num_channels = shape[-1]//2
    n_points = shape[1]*shape[2]
    real_pdt = y_pred[:,:,:,num_channels:]
    imag_pdt = y_pred[:,:,:,:num_channels]
    real_true = y_true[:,:,:,num_channels:]
    imag_true = y_true[:,:,:,:num_channels]
    print(tf.sqrt(tf.reduce_sum((real_pdt-real_true)**2+(imag_pdt-imag_true)**2)/n_points))
    return tf.sqrt(tf.reduce_sum((real_pdt-real_true)**2+(imag_pdt-imag_true)**2)/n_points)

np.random.seed(1)
x_train = np.random.rand(61,64,64,4)
y_train = np.random.rand(61,64,64,16)

tf.keras.backend.clear_session()
input_shapes = x_train.shape[1:]

InputTensor = keras.Input(shape=(64,64,4))
conv1 = complexnn.conv.ComplexConv2D(64,(3,3),padding='same')(InputTensor)
Output = AmplitudeMaxout_piece(8)(conv1)
model = keras.Model(inputs=InputTensor,outputs=Output)
model.compile(optimizer=keras.optimizers.Adam(),loss=ComplexRMS)
model.fit(x_train,y_train,epochs=25)
