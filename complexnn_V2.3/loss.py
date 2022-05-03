# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:25:07 2021

@author: benzener
"""

import tensorflow as tf
from .utils_test import get_realpart, get_imagpart
import tensorflow.keras.backend as K
import numpy as np
import scipy.signal as Signal
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError

PI = tf.constant(np.pi)

def ComplexRMS(y_true, y_pred):
    y_true, y_pred = _precheck(y_true, y_pred)
    return tf.sqrt(tf.reduce_mean(_get_square_error(y_true, y_pred)))

def ComplexMSE(y_true, y_pred):
    '''
    ComplexMSE esitmates the mean-squared error according to the
    real part and imaginary part one-by-one.

    Args:
        y_true: the target with shape [N,H,W,C], where C must be
            a multiple of 2 due to complex narture.
        y_pred: the predicted data with the same shape as y_true.
    
    Returns:
        a Tensor scalar, the mean value of the whole batch.

    '''
    y_true, y_pred = _precheck(y_true, y_pred)
    return tf.reduce_mean(_get_square_error(y_true, y_pred))

def ComplexMAE(y_true, y_pred):
    '''
    ComplexMAE esitmates the mean-absolute error according to the
    real part and imaginary part one-by-one.

    Args:
        y_true: the target with shape [N,H,W,C], where C must be
            a multiple of 2 due to complex narture.
        y_pred: the predicted data with the same shape as y_true.
    
    Returns:
        a Tensor scalar, the mean value of the whole batch.

    '''
    y_true, y_pred = _precheck(y_true, y_pred)
    mae = MeanAbsoluteError()
    return mae(y_true, y_pred)


# def _feature_size(x):
#     # to avoid the first unknown batch size, cannot use reshape(x,[-1]) to determine the nb of elements
#     shape = x.get_shape().as_list()
#     n = 1
#     for dim in shape[1:]:
#         n = n * dim
#     return tf.divide(tf.cast(n,dtype=tf.float32),2)


def _get_square_error(y_true, y_pred):
    '''
    Compute the sqaured residual based on the two channel, real
    and imaginary part.

    Args:
        y_true: the target with shape [N,H,W,C], where C must be
            a multiple of 2 due to complex narture.
        y_pred: the predicted data with the same shape as y_true.
    
    Returns:
        a Tensor of size N represents squared error.

    '''
    real_pdt = get_realpart(y_pred)
    imag_pdt = get_imagpart(y_pred)
    real_true = get_realpart(y_true)
    imag_true = get_imagpart(y_true)
    # squared_error = (y_hat_real - y_real)**2 + (y_hat_imag - y_imag)**2
    result = tf.add(tf.pow(tf.subtract(real_pdt,real_true),2),tf.pow(tf.subtract(imag_pdt,imag_true),2))
    return result

def _precheck(y_true,y_pred):
#    if y_true.get_shape().is_fully_defined():
#        if not y_true.get_shape().is_compatible_with(y_pred.get_shape()):
#            _f1()
#        else:
#            _f2()
#    print(y_true.get_shape().is_compatible_with(y_pred.get_shape()))
    if not tf.is_tensor(y_true):
        y_true = tf.constant(y_true)
    if not tf.is_tensor(y_pred):
        y_pred = tf.constant(y_pred)
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

# def _envelope_detection(signal): 
#     shape = signal.shape
#     if shape[-1]%2:
#         envelope = np.abs(Signal.hilbert(signal, axis=1))
#     else:
#         channel = shape[-1]//2
#         envelope = np.sqrt(signal[:,:,:,channel:]**2 + signal[:,:,:,channel:]**2)
#     ratio = np.max(envelope, axis=(1,2,3),keepdims=True)
#     return (envelope/ratio).astype(np.float32)

# # @tf.function(input_signature=[tf.TensorSpec(None,dtype=tf.float32),tf.TensorSpec(None,dtype=tf.float32)])
# def SSIM(y_true,y_pred):
#     envelope_true = tf.py_function(_envelope_detection,[y_true],tf.float32)
#     envelope_pred = tf.py_function(_envelope_detection,[y_pred],tf.float32)
#     tf.print(1 - tf.reduce_mean(tf.image.ssim(envelope_pred,envelope_true,max_val=1,filter_size=7)))
#     return 1 - tf.reduce_mean(tf.image.ssim(envelope_pred,envelope_true,max_val=1,filter_size=7))

def _normalization(signal):
    '''
    normalize the input signal.

    Args:
        signal: baseband signal with shape [N,H,W,C], where C must
            be a multiple of 2. Future work may support to RF data.
            2022/04/25 ver
    Returns:
        a normalized Tensor

    '''
    if signal.shape[-1]%2:
        # the real-valued is normalized by itself maximum
        normalized_target = tf.abs(signal)

    else:
        # the complex-valued is normalized by modulus
        real = get_realpart(signal)
        imag = get_imagpart(signal)
        normalized_target = (real**2 + imag**2)**0.5
    ratio = tf.reduce_max(normalized_target,axis=(1,2,3),keepdims=True)
    return signal/ratio

def _angle(x):
    return tf.math.angle(tf.complex(get_realpart(x),get_imagpart(x)))
 
def _envelope_detection(signal):
    '''
    envelope detection. Convert BB signal to envelope signal.

    Args:
        signal: baseband signal with shape [N,H,W,C], where C must
            be a multiple of 2. Future work may support to RF data.
            2022/04/25 ver
    Returns:
        a normalized Tensor with only real-valued channel.

    '''
    envelope = (get_realpart(signal)**2 + get_imagpart(signal)**2)**0.5
    return _normalization(envelope)
   
def _SSIM_core(y_true, y_pred, func):
    '''
    SSIM esitmation.

    Args:
        y_true: the target with shape [N,H,W,C], where C must be
            a multiple of 2 due to complex narture.
        y_pred: the predicted data with the same shape as y_true.
        func: a Tensor function either ssim or ms-ssim
    
    Returns:
        a Tensor scalar, the (mean) value of the whole batch.

    '''
    if y_pred.shape[-1]%2:
        y_pred = _normalization(y_pred)
        return 1 - tf.reduce_mean(func(y_pred,y_true,max_val=2,filter_size=7))
    else:
        # envelope_true = _envelope_detection(y_true)
        # envelope_pred = _envelope_detection(y_pred)
        # return 1 - tf.reduce_mean(func(envelope_pred,envelope_true,max_val=1,filter_size=7))
        y_pred = _normalization(y_pred)
        return 1 - tf.reduce_mean(func(y_pred+1, y_true+1, max_val=2,filter_size=7))
        # pi = 2*tf.acos(0.)
        # angle_ssim = 1 - tf.reduce_mean(func(_angle(y_pred),_angle(y_true),max_val=2*pi,filter_size=7))
        # envelope_ssim = 1 - tf.reduce_mean(func(envelope_pred,envelope_true,max_val=1,filter_size=7))
        # return (angle_ssim + envelope_ssim)/2
        
        
def SSIM(y_true,y_pred):
    y_true, y_pred = _precheck(y_true, y_pred)
    return _SSIM_core(y_true, y_pred, tf.image.ssim)

def MS_SSIM(y_true,y_pred):
    y_true, y_pred = _precheck(y_true, y_pred)
    return _SSIM_core(y_true, y_pred, tf.image.ssim_multiscale)
    
def SSIM_MSE(y_true,y_pred):
    y_true, y_pred = _precheck(y_true, y_pred)
    ratio = 0.002
    if y_pred.shape[-1]%2:
        mse = MeanSquaredError()
        mse = mse(y_true, y_pred)
    else:
        mse = MeanSquaredError()
        mse = mse(PI + _angle(y_true),PI + _angle(y_pred))
        # mse = ComplexMSE(y_true, y_pred)
    ssim = SSIM(y_true,y_pred)
    return ratio*ssim + mse
    # return ssim*mse

get_custom_objects().update({'SSIM': SSIM,
                             'MS_SSIM': MS_SSIM,
                             'SSIM_MSE': SSIM_MSE})







