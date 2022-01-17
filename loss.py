# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:25:07 2021

@author: benzener
"""

import tensorflow as tf
from .utils import get_realpart, get_imagpart


def ComplexRMS(y_true, y_pred):
    y_true, y_pred = _precheck(y_true, y_pred)
    n = _feature_size(y_pred)
    return tf.sqrt(tf.reduce_sum(_get_square_error(y_true, y_pred)) /n)


def ComplexMSE(y_true, y_pred):
    y_true, y_pred = _precheck(y_true, y_pred)
    n = _feature_size(y_pred) 
    return tf.reduce_sum(_get_square_error(y_true, y_pred)) / n


def ComplexMAE(y_true, y_pred):
    y_true, y_pred = _precheck(y_true, y_pred)
    n = _feature_size(y_pred)
    return tf.reduce_sum(tf.sqrt(_get_square_error(y_true, y_pred))) / n


def _feature_size(x):
    # to avoid the first unknown batch size, cannot use reshape(x,[-1]) to determine the nb of elements
    shape = x.get_shape().as_list()
    n = 1
    for dim in shape[1:]:
        n = n * dim
    return tf.cast(n,dtype=tf.float32)/2

def _get_square_error(y_true, y_pred):
    print('ture',tf.shape(y_true),'pred',tf.shape(y_pred))
    real_pdt = get_realpart(y_pred)
    imag_pdt = get_imagpart(y_pred)
    real_true = get_realpart(y_true)
    imag_true = get_imagpart(y_true)
    return (real_pdt-real_true)**2 + (imag_pdt-imag_true)**2


def _precheck(x,y):
    return tf.cast(x,dtype=tf.float32), tf.cast(y,dtype=tf.float32)