# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:25:07 2021

@author: benzener
"""

import tensorflow as tf
from .utils_test import get_realpart, get_imagpart
import tensorflow.compat.v1.keras.backend as K


def ComplexRMS(y_true, y_pred):
    y_true, y_pred = _precheck(y_true, y_pred)
    # n = _feature_size(y_pred)
    return tf.sqrt(tf.reduce_sum(_get_square_error(y_true, y_pred),axis=1))
'''
n應為complex channel數量 即shape[-1]//2
在此尚未修改 仍為2*n
'''
def ComplexMSE(y_true, y_pred):
    y_true, y_pred = _precheck(y_true, y_pred)
#    n = _feature_size(y_pred) 
    return tf.reduce_sum(_get_square_error(y_true, y_pred),axis=1)

def ComplexMAE(y_true, y_pred):
    y_true, y_pred = _precheck(y_true, y_pred)
    # n = _feature_size(y_pred)
    return tf.reduce_sum(tf.sqrt(_get_square_error(y_true, y_pred)),axis=1)


# def _feature_size(x):
#     # to avoid the first unknown batch size, cannot use reshape(x,[-1]) to determine the nb of elements
#     shape = x.get_shape().as_list()
#     n = 1
#     for dim in shape[1:]:
#         n = n * dim
#     return tf.divide(tf.cast(n,dtype=tf.float32),2)


def _get_square_error(y_true, y_pred):
    real_pdt = get_realpart(y_pred)
    imag_pdt = get_imagpart(y_pred)
    real_true = get_realpart(y_true)
    imag_true = get_imagpart(y_true)
    result = tf.add(tf.pow(tf.subtract(real_pdt,real_true),2),tf.pow(tf.subtract(imag_pdt,imag_true),2))
    return result

def _precheck(y_true,y_pred):
#    if y_true.get_shape().is_fully_defined():
#        if not y_true.get_shape().is_compatible_with(y_pred.get_shape()):
#            _f1()
#        else:
#            _f2()
#    print(y_true.get_shape().is_compatible_with(y_pred.get_shape()))
    if y_true.shape.is_fully_defined():
        if not y_true.shape.is_compatible_with(y_pred.shape):
            raise ValueError('Expected shape is ' + str(K.shape(y_pred)[1:]) + ' but get ' + str(y_true.shape[1:]))
#        
#    def _raise_error():
#        raise ValueError('Expected shape is ' + str(K.shape(y_pred)[1:]) + ' but get ' + str(y_true.shape[1:]))
#    def _pass():
#        return None
#    print(tf.reduce_all(tf.equal(tf.shape(y_true),tf.shape(y_pred))))
#    if y_true.shape.is_compatible_with(y_pred.shape):
#        print('equal')
#    tf.cond(tf.reduce_all(tf.equal(tf.shape(y_true),tf.shape(y_pred))),
#            true_fn = _pass,
#            false_fn = _raise_error
#            )
#    tf.cond(tf.reduce_all(tf.equal(tf.shape(y_true),tf.shape(y_pred))),
#            true_fn = _pass,
#            false_fn = _raise_error
#            )
#    print(tf.cast(y_true,dtype=tf.float32), tf.cast(y_pred,dtype=tf.float32))
#    return tf.cast(y_true,dtype=tf.float32), tf.cast(y_pred,dtype=tf.float32)

#    print(flag)
#    if flag:
#        return tf.cast(y_true,dtype=tf.float32), tf.cast(y_pred,dtype=tf.float32)
#
#    return y_true, y_pred

#    if not tf.reduce_all(tf.equal(K.shape(y_true),K.shape(y_pred))):

#    result = tf.cond(tf.reduce_all(tf.equal(K.shape(y_true),K.shape(y_pred))),
#                true_fn=_f1(),
#                false_fn=_f2(K.shape(y_true),K.shape(y_pred))))
#    if not y_true.get_shape().is_compatible_with(y_pred.get_shape()):
#        print('wrong')
#        print(tf.ensure_shape(y_true.get_shape(),y_pred.get_shape()))
    return tf.cast(y_true,dtype=tf.float32), tf.cast(y_pred,dtype=tf.float32)

        







