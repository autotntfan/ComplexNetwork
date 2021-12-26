# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:25:07 2021

@author: benzener
"""

import tensorflow as tf

class ComplexLoss:
    
    def __init__(self):
        pass
        

    def ComplexRMS(self, y_true, y_pred):
        assert tf.shape(y_true) == tf.shape(y_pred)
        n = self.feature_size(y_pred)
        return tf.sqrt(tf.reduce_sum(self.get_square_error(y_true, y_pred)) /n)
    

    def ComplexMSE(self, y_true, y_pred):
        assert tf.shape(y_true) == tf.shape(y_pred)
        n = self.feature_size(y_pred) 
        return tf.reduce_sum(self.get_square_error(y_true, y_pred)) / n
    

    def ComplexMAE(self, y_true, y_pred):
        assert tf.shape(y_true) == tf.shape(y_pred)
        n = self.feature_size(y_pred)
        return tf.reduce_sum(tf.sqrt(self.get_square_error(y_true, y_pred))) / n
    
    def get_real_and_imag(self, x):
        shape = x.get_shape().as_list()
        if shape[-1]%2:
            raise ValueError(f"nb of imaginary part isn't equal to that of real part")
        num_channels = shape[-1]//2
        real = x[:,:,:,num_channels:]
        imag = x[:,:,:,:num_channels]
        return real, imag
    
    def feature_size(self, x):
        shape = x.get_shape().as_list()
        n = 1
        for dim in shape[1:]:
            n = n * dim
        return tf.cast(n,dtype=tf.float32)/2
    
    def get_square_error(self, y_true, y_pred):
        real_pdt, imag_pdt = self.get_real_and_imag(y_pred)
        real_true, imag_true = self.get_real_and_imag(y_true)
        return (real_pdt-real_true)**2 + (imag_pdt-imag_true)**2