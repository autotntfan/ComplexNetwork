# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:44:42 2021

@author: benzener
"""

import tensorflow as tf
from .utils import get_realpart, get_imagpart
from keras.layers import Layer

def modReLU(x):
    b = tf.Variable(-4., trainable=True)
    def return_value():
        real = get_realpart(x)
        imag = get_imagpart(x)
        modulus = tf.sqrt(real**2 + imag**2)
        phase = tf.atan2(imag, real)
        cond = (tf.add(modulus, b) >= 0)
        modulus = tf.where(cond, modulus, 0)
        real = modulus*tf.cos(phase)
        imag = modulus*tf.sin(phase)
        return tf.concat([real, imag], axis=-1)
    return return_value()

def CReLU(x):
    real = get_realpart(x)
    imag = get_imagpart(x)
    return tf.concat([tf.nn.relu(real),tf.nn.relu(imag)], axis=-1)

def zReLU(x):
    real = get_realpart(x)
    imag = get_imagpart(x)
    cond = (real >= 0) & (imag >= 0)
    real = tf.where(cond, real, 0.)
    imag = tf.where(cond, imag, 0.)
    return tf.concat([real,imag], axis=-1)


class AmplitudeMaxout(Layer):
    
    def __init__(self, num_pieces, name='AMU',**kwargs):
        super().__init__(**kwargs)
        self.__name__ = name
        self.num_pieces = num_pieces
        

    def call(self, x, axis=None):
        shape = x.get_shape().as_list()
        if axis is None:
            axis = -1
            shape[0] = -1
        if shape[axis]%2:
            raise ValueError(f'nb of real/imaginary channel are inequivalent')
        num_channels = shape[-1]//2
        self.num_units = num_channels//self.num_pieces
        if num_channels%self.num_pieces:
            self.num_units += 1
            num_padding = self.num_pieces - num_channels%self.num_pieces
            padding_size = tf.concat([tf.shape(x)[:-1],tf.constant([num_padding])],axis=-1)
            zero_padding = tf.zeros(padding_size)        
        shape[axis] = self.num_units
        exp_shape = shape + [self.num_pieces]
        real_part = x[:,:,:,:num_channels]
        imag_part = x[:,:,:,num_channels:]    
        if num_channels%self.num_pieces:
            real_part = tf.concat([real_part,zero_padding], axis=-1)
            imag_part = tf.concat([imag_part,zero_padding], axis=-1)
        real_part = tf.reshape(real_part, exp_shape)
        imag_part = tf.reshape(real_part, exp_shape)
        real_part, imag_part = self.return_AMU(real_part, imag_part, exp_shape)
        tf.print(tf.concat([real_part,imag_part],axis=-1))
        return tf.concat([real_part,imag_part],axis=-1)         

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape[-1] = 2*self.num_units
        return tuple(shape)
    
    def get_config(self):
        base_config = super().get_config()
        config = {'units': self.num_units,
                  'pieces': self.num_pieces}
        return dict(list(base_config.items()) + list(config.items()))
    
    def return_AMU(self, real_part, imag_part ,expand_shape):

        modulus = real_part**2 + imag_part**2
        expand_modulus = tf.reshape(modulus, expand_shape)    
        cond = tf.equal(expand_modulus,tf.reduce_max(expand_modulus,axis=-1,keepdims=True))
        real_part = tf.reduce_max(real_part*tf.cast(cond,dtype=tf.float32),axis=-1)
        imag_part = tf.reduce_max(imag_part*tf.cast(cond,dtype=tf.float32),axis=-1)    
        return real_part, imag_part

    

