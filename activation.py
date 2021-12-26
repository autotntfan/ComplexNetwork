# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:44:42 2021

@author: benzener
"""

import tensorflow as tf

class ComplexActivation:
    
    def __init__(self):
        pass
    
    def modReLU(self, x):
        b = tf.Variable(-4., trainable=True)
        def return_value():
            real, imag = self.get_real_and_imag(x)
            modulus = tf.sqrt(real**2 + imag**2)
            phase = tf.atan2(imag, real)
            cond = (tf.add(modulus, b) >= 0)
            modulus = tf.where(cond, modulus, 0)
            real = modulus*tf.cos(phase)
            imag = modulus*tf.sin(phase)
            return tf.concat([real, imag], axis=-1)
        return return_value()
    
    def CReLU(self, x):
        real, imag = self.get_real_and_imag(x)
        return tf.concat([tf.nn.relu(real),tf.nn.relu(imag)], axis=-1)
    
    def zReLU(self, x):
        real, imag = self.get_real_and_imag(x)
        cond = (real >= 0) & (imag >= 0)
        real = tf.where(cond, real, 0.)
        imag = tf.where(cond, imag, 0.)
        return tf.concat([real,imag], axis=-1)
    
    def get_real_and_imag(self, x):
        shape = x.get_shape().as_list()
        if shape[-1]%2:
            raise ValueError(f"nb of imaginary part isn't equal to that of real part")
        num_channels = shape[-1]//2
        real = x[:,:,:,num_channels:]
        imag = x[:,:,:,:num_channels]
        return real, imag
