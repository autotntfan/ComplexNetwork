# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 23:28:49 2022

@author: benzener
"""


import os
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
    from baseband.utils.info import get_sampling_rate, get_data
    sys.path.remove(addpath)
else:
    from ..setting import constant
    from .info import get_sampling_rate, get_data
import scipy.signal      as Signal
import numpy as np
import cv2

def convert_to_complex(inputs):
    '''
    Convert real-valued input to complex-valued type array.
        Args:
            inputs: Numpy array, real-valued or complex-type.
        Return:
            complex-valued array with the same dimension of inputs.
        
    Example:
        input = RF data, output = complex type BB data (a+bi)
        
        input = 2-channel real type BB data, output = complex type BB data (a+bi)
        
        input = complex type BB data, output = complex type BB data (a+bi)
    '''
    if np.iscomplex(inputs).all():
        return inputs
    shape = inputs.shape
    rank = inputs.ndim
    if rank == 2:
        return Signal.hilbert(inputs, axis=0)
    elif rank == 3:
        # only allow [H,W,C]
        if shape[-1]%2:
            # real-valued signal needs hilbert transform
            return Signal.hilbert(inputs, axis=0)
        else:
            real, imag = split_complex(inputs)
            return real + 1j*imag
    elif rank == 4:
        if shape[-1]%2:
            # real-valued signal needs hilbert transform
            return Signal.hilbert(inputs, axis=1)
        else:
            real, imag = split_complex(inputs)
            return real + 1j*imag
    else:
        raise ValueError(f'Unrecognized complex array with shape {shape}')
        

def convert_to_real(inputs):
    '''
    Convert complex data type input to 2-channel real type array. If input is real data type, this
    function returns intput itself.
        Args:
            inputs: Numpy array, real-valued or complex-type.
        Return:
            real-valued array with the same dimension of inputs.
        
    Example:
        input = RF data, output = real type RF data
        
        input = complex-valued data, output = 2-channel real type BB data
        
        input = 2-channel real type BB data, output = 2-channel real type BB data
    '''
    if np.isreal(inputs).all():
        return inputs
    shape = inputs.shape
    rank = inputs.ndim
    if rank == 2:
        # complex dtype [H,W] -> [H,W,2] real dtype
        real, imag = split_complex(inputs)
        real = np.expand_dims(real, axis=-1)
        imag = np.expand_dims(imag, axis=-1)
        return np.concatenate((real,imag),axis=-1)
    elif rank in {3,4}:
        # complex dtype only allow [H,W,C]
        real, imag = split_complex(inputs)
        return np.concatenate((real,imag),axis=-1)
    else:
        raise ValueError(f'Unrecognized complex array with shape {shape}')
        
def bbdemodulate(signals, inds):
    '''
    Baseband demodulation
    '''
    if np.iscomplex(signals).any():
        # complex dtype
        return signals
    # real dtype
    try:
        N = len(inds)
    except TypeError:
        N = len(np.asarray([inds]))
    if N == 1:
        if signals.shape[-1]%2 == 0:
            # BB data [H,W,C]
            real, imag = split_complex(signals)
            return real + 1j*imag
        # [H,W,C], [H,W]
        fc = np.squeeze(get_data(inds, 'f0'))
        fs = get_sampling_rate(signals, inds)
        t = np.arange(0,signals.shape[0])/fs
        t = t.reshape((signals.shape[0],1))
        lpf = Signal.firwin(constant.FIRORDER, fc/(fs/2))
        twoDsignal = reduce_dim(signals)*np.exp(-1j*2*np.pi*fc*t)
        assert twoDsignal.ndim == 2
        assert lpf.ndim ==1
        twoDsignal = np.apply_along_axis(Signal.convolve, 0, twoDsignal, lpf, 'same')
        # [H,W,C]
        if signals.shape[-1] in {1,2}:
            # RF data [H,W,C]
            return twoDsignal.reshape(signals.shape)  
        else:
            # [H,W] only RF
            return twoDsignal
    else:
        # [N,H,W,C], [N,H,W]
        # BB data [N,H,W,C]
        if signals.ndim == 4 and signals.shape[-1]%2 == 0:
            real, imag = split_complex(signals)
            return real + 1j*imag
        
        bbsignal = np.zeros_like(signals, dtype=np.complex64)
        for ii in range(len(signals)):
            fc = np.squeeze(get_data(inds[ii], 'f0'))
            fs = get_sampling_rate(signals[ii], inds[ii])
            t = np.arange(0,signals.shape[1])/fs
            t = t.reshape((-1,1))
            lpf = Signal.firwin(constant.FIRORDER, fc/(fs/2))
            twoDsignal = reduce_dim(signals[ii])*np.exp(-1j*2*np.pi*fc*t)
            assert twoDsignal.ndim == 2
            assert lpf.ndim ==1
            twoDsignal = np.apply_along_axis(Signal.convolve, 0, twoDsignal, lpf, 'same')
            if signals.ndim == 4:
                # RF data [N,H,W,C]
                bbsignal[ii] = twoDsignal.reshape(signals.shape[1:-1] + (1,))
            else:
                bbsignal[ii] = twoDsignal
            if ii%10 == 0:
                print(f'Baseband demodulating ... >> {ii}/{len(signals)}')
        return bbsignal
            
        
        
def split_complex(x):
    '''
    Split complex-valued type or complex-type array to real and imaginary part
        Args:
            x: Numpy array, complex-valued or complex-type.
        Returns:
            Two numpy array represent real and imaginary part respectively with
            the same dimension of inputs.
    
    Example:
        input = complex type BB data, output = Re{BB}, Im{BB}
        i.e. a+bi -> a,b
        
        input = real type RF data, output = Re{BB}, Im{BB}
        i.e. z -> a,b
        
        input = 2-channel real type BB data, output = first channel, second channel
    '''
    # for complex-valued type array
    if np.iscomplex(x).any():
        return np.real(x), np.imag(x)
    shape = x.shape
    rank = x.ndim
    if rank == 2:
        x = convert_to_complex(x)
        return np.real(x), np.imag(x)
    elif rank == 3:
        # for complex-type array
        # only allow [H,W,C]
        if shape[-1]%2:
            x = convert_to_complex(x)
            return np.real(x), np.imag(x)
        else:
            return x[:,:,:shape[-1]//2], x[:,:,shape[-1]//2:]
    elif rank == 4:
        if shape[-1]%2:
            x = convert_to_complex(x)
            return np.real(x), np.imag(x)
        else:
            return x[:,:,:,:shape[-1]//2], x[:,:,:,shape[-1]//2:]
    else:
        raise ValueError(f'Unrecognized complex array with shape {shape}')   

def normalization(inputs):
    '''
    Limit input in the range of [-1,1] for coherent signal, or [0,1]
    for incoherent signal.
    
    Args:
        inputs: ndarray, in the shape of [N,H,W,C], [H,W,C], [H,W]
    
    Return:
        ndarray, max value = orignal value/abs(max value) along each data
    
    '''
    shape = inputs.shape
    rank = inputs.ndim
    if rank == 2:
        # [H,W] in real or complex type
        return inputs/np.max(np.abs(inputs))
    elif rank == 3:
        # only [H,W,C], [N,H,W] is NOT available
        if shape[-1]%2:
            # real type array
            return inputs/np.max(np.abs(inputs))
        else:
            # complex type array
            real = inputs[:,:,:shape[-1]//2]
            imag = inputs[:,:,shape[-1]//2:]
            modulus = np.sqrt(real**2 + imag**2)
            return inputs/np.max(modulus)
    elif rank == 4:
        if shape[-1]%2:
            # real type array, e.g. [N,H,W,1]
            return inputs/np.max(np.abs(inputs), axis=(1,2,3), keepdims=True)
        else:
            # complex type array, e.g. [N,H,W,2]
            real = inputs[:,:,:,:shape[-1]//2]
            imag = inputs[:,:,:,shape[-1]//2:]
            modulus = np.sqrt(real**2 + imag**2)
            return inputs/np.max(modulus, axis=(1,2,3), keepdims=True)
    else:
        raise ValueError(f"Unrecognized data type with shape {shape}")
        
def check_data_range(x):
    print(f"The maximum value is {np.max(x)} and the"
          " minimum value is {np.min(x)}")

def precheck_dim(inputs):
    # expand dim to 4 -> [1,H,W,C]
    if inputs.ndim == 4:
        if inputs.shape[0] != 1:
            raise ValueError('Only support one image')
    elif inputs.ndim == 3:
        # [H,W,C] -> [1,H,W,C]
        inputs = np.expand_dims(inputs, axis=0)
    elif inputs.ndim == 2:
        # [H,W] -> [1,H,W,1]
        inputs = inputs.reshape((1,) + inputs.shape + (1,))
    assert inputs.ndim == 4
    return inputs

def reduce_dim(inputs):
    # reduce the dimension of one image to [H,W]
    inputs = precheck_dim(inputs)
    output_shape = inputs.shape[1:-1]
    return inputs.reshape(output_shape)

def envelope_detection(signal, gain=None):
    '''
    Detect envelope, where gain is equivalent to dynamic range. However, actually
    gain is not equal to DR always. It should be envelope_detection(self, signal, DR, gain).
    
    *The log(0) is undefined, hence that is added by 1e-16, i.e. at least -320 dB.
    
    *Before log compression, data has been normalized since we implement SSIM loss which
    has to limit prediction and reference in the same value range, e.g. [-1,1], but model
    prediction DO NOT pass through loss that value range is not in [-1,1]. Accordingly, here 
    we have to implement normalization to preserve its range is valid and further comparsion
    to be fair.
        Args:
            signal: Numpy array with shape [N,H,W,C], [H,W], or [H,W,C], it could be real or complex type. 
            gain: scalar, gain for log compression.
        Return:
            Numpy array in dB scale if DR exists, otherwise linear scale.
    
    '''
    signal = normalization(signal)
    envelope = np.abs(convert_to_complex(signal))
    if gain is None:
        return envelope
    else:
        if signal.ndim == 4:    
            return np.squeeze(20*np.log10(envelope/np.max(envelope,axis=(1,2,3),keepdims=True) + constant.EPS) + gain)
        else:
            return np.squeeze(20*np.log10(envelope/np.max(envelope) + constant.EPS) + gain)
    
def projection(signal, gain=None, direction='lateral', vmin=None):
    '''
    Axial or lateral projection of signal
    Args:
        signal: ndarray, target
        gain: scalar, gain for log compression.
        direction: string, only allow 'lateral' and 'axial'
        vmin: the minimum value of projection,
            i.e. vmin=0, gain=60 then value<0 would be forced to 0
            and the max value is 60
    Return:
        1-D/2-D projection data along axial or lateral determined
        by input sequence.
    '''
    if direction not in {'lateral','axial'}:
        raise ValueError("direction only along 'lateral' or 'axial' ")
    if signal.ndim == 4:
        # [N,H,W,C]
        axis = 1 if direction == 'lateral' else 2
    elif signal.ndim < 2 or signal.ndim > 4:
        raise ValueError(f'Unsupport dimension {signal.ndim}')
    else:
        # [H,W,C] or # [H,W]
        axis = 0 if direction == 'lateral' else 1
    if gain is None:
        # directly applied projection without log compression
        return np.max(signal, axis, initial=vmin)
    else:
        # do log compression depending on DR
        outputs = np.max(envelope_detection(signal,gain), axis, initial=vmin)
        return np.squeeze(outputs)
    
def angle(signal):
    '''
    Compute the angle (phase) for complex value.
        Args:
            signal: Numpy array, complex-valued or real-valued type.
        Return:
            Unwrapping angle
    '''
    complex_signal = convert_to_complex(signal)
    wrapped_angle = np.angle(complex_signal)
    if wrapped_angle.ndim == 4:
        return np.apply_over_axes(np.unwrap, wrapped_angle, [1,2])
    else:
        return np.apply_over_axes(np.unwrap, wrapped_angle, [0,1])
    
def focusing(img, ratio=0.05):
    '''
    In order to remove artifacts around edge.
        Args:
            img: Numpy array, displayed images.
            ratio: A decimal, clipping ratio of image.
        Return:
            Pruned numpy arrays.
    '''
    if ratio <= 0 or ratio >= 1:
        raise ValueError('ratio is out of boundary')
    shape = img.shape
    if img.ndim == 4:
        H, W = round(shape[1]*ratio), round(shape[2]*ratio)
        return img[:,H:-H,W:-W,:]
    elif img.ndim == 3:
        H, W = round(shape[0]*ratio), round(shape[1]*ratio)
        return img[H:-H,W:-W,:]
    elif img.ndim == 2:
        H, W = round(shape[0]*ratio), round(shape[1]*ratio)
        return img[H:-H,W:-W]
    else:
        raise ValueError(f'Unrecognized complex array with shape {shape}')
        
def downsampling(signal, factor, direction='lateral'):
    '''
    Downsample signal by a factor along a specific direction.
    
        Args:
            signal: ndarray with shape [N,W,H,C], [H,W,C], [N,H,W], or [H,W].
            factor: int, decimation factor.
            direction: string, it can be 'lateral', 'axial', or 'both'
        Return:
            ndarray
        e.g. input size = [10,257,257,2], factor = 2, direction='both'
            output size = [10,129,129,2]
            
    '''
    if direction not in {'lateral','axial','both'}:
        raise ValueError('Direction only allows lateral, axial, and both')
    if signal.ndim == 4:
        # [N,H,W,C]
        if direction == 'lateral':
            return signal[:,:,::factor,:]
        elif direction == 'axial':
            return signal[:,::factor,:,:]
        else:
            return signal[:,::factor,::factor,:]
    elif signal.ndim == 3:
        if signal.shape[-1] in {1,2}:
            # [H,W,C]
            if direction == 'lateral':
                return signal[:,::factor,:]
            elif direction == 'axial':
                return signal[::factor,:,:]
            else:
                return signal[::factor,::factor,:]
        else:
            # [N,H,W]
            if direction == 'lateral':
                return signal[:,:,::factor]
            elif direction == 'axial':
                return signal[:,::factor,:]
            else:
                return signal[:,::factor,::factor]
    elif signal.ndim == 2:
        # [H,W]
        if direction == 'lateral':
            return signal[:,::factor]
        elif direction == 'axial':
            return signal[::factor,:]
        else:
            return signal[::factor,::factor]
    else:
        raise ValueError('Only support 2D,3D,and 4D signal')

def lowbeamspacing(signal, factor, direction='lateral'):
    '''
    Combine downsampling and upsampling. The interpolation way is bicubic supplied by opencv.
        Args:
            signal: ndarray with shape [N,W,H,C], [H,W,C], [N,H,W], or [H,W].
            factor: int, decimation (upsampling) factor.
            direction: string, it can be 'lateral', 'axial', or 'both'
        Return:
            ndarray, same shape as the input
        e.g. input size = [10,257,257,2], factor = 2, direction='both'
            output size = [10,257,257,2]
            
    '''
    if factor == 1:
        return signal
    elif not isinstance(factor, int):
        raise TypeError("factor only allows int data type")
    dsignal = downsampling(signal, factor, direction) # downsampling signal
    usignal = np.zeros_like(signal) # upsampling signal
    if signal.ndim == 4:
        # [N,H,W,C]
        N, H, W = signal.shape[:-1]
        for ii in range(N):
            usignal[ii] = cv2.resize(dsignal[ii],(W,H),interpolation=cv2.INTER_CUBIC)
        return usignal
    elif signal.ndim == 3:
        if signal.shape[-1] in {1,2}:
            # [H,W,C]
            H, W = signal.shape[:-1]
            return cv2.resize(dsignal,(W,H),interpolation=cv2.INTER_CUBIC)
        else:
            # [N,H,W]
            H, W  = signal.shape[1:]
            for ii in range(N):
                usignal[ii] = cv2.resize(dsignal[ii],(W,H),interpolation=cv2.INTER_CUBIC)
            return usignal
    elif signal.ndim == 2:
        # [H,W]
        H, W = signal.shape
        return cv2.resize(dsignal,(W,H),interpolation=cv2.INTER_CUBIC)
    else:
        raise ValueError('Only support 2D,3D,and 4D signal')