# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 23:28:49 2022

@author: benzener
"""


import os
if __name__ == '__main__':
    import sys
    currentpath = os.getcwd()
    addpath = os.path.dirname(os.path.dirname(currentpath))
    if addpath not in sys.path:
        sys.path.append(addpath)
    from baseband.setting import constant
    from baseband.utils.info import get_sampling_rate, get_data, progressbar, isrf, isbb
    sys.path.remove(addpath)
else:
    from ..setting import constant
    from .info import get_sampling_rate, get_data, progressbar, isrf, isbb
import scipy.signal      as Signal
import numpy as np
import cv2


        
def split_complex(x, hasN=True):
    '''
    Split complex value (2-channel real format or pure complex format) into real and imaginary part.
    If the "x" is RF data, x will be firstly convert to BB data and then splitted into real and imaginary part.
    Allow data shape includes:
                complex format (BB data)        real format
        4-D             NHWC                  NHWC (RF or BB data)
        3-D            NHW.HWC          NHW (RF data).HWC (RF or BB data)
        2-D             NH.HW            NH (RF data).HW (RF data)
        1-D               H                      H (RF data)
        
        Args:
            x: Numpy array, complex-valued or complex-type.
            hasN: Boolean, whether the first dimension represents amount N.
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
        # return original shape
        return np.real(x), np.imag(x)
    shape = x.shape
    rank = x.ndim
    if rank not in {1,2,3,4}:
        raise ValueError(f'Unrecognized complex array with shape {shape}')
    if isrf(x, hasN):
        # [N,H,W,1], [N,H,W], [H,W,1], [N,H], [H,W], [H]
        x = convert_to_complex(x, hasN)
    if rank in {1,2}:
        # [H], [H,W], [N,H]
        return np.real(x), np.imag(x)
    if rank == 3:
        if shape[-1]%2 == 0 and not hasN:
            # [H,W,2] -> BB data
            return x[:,:,:shape[-1]//2], x[:,:,shape[-1]//2:]
        else:
            # [N,H,W] -> RF data
            # [H,W,1] -> RF data
            return np.real(x), np.imag(x)         
    if rank == 4:
        if shape[-1]%2:
            # [N,H,W,C] -> RF data
            return np.real(x), np.imag(x)
        else:
            # [N,H,W,C] -> BB data
            return x[:,:,:,:shape[-1]//2], x[:,:,:,shape[-1]//2:]

        
def convert_to_complex(inputs, hasN=True):
    '''
    This function is used to convert data format from real input to complex array.
    Allow data shape includes:
                complex format (BB data)        real format
        4-D             NHWC                  NHWC (RF or BB data)
        3-D            NHW.HWC          NHW (RF data).HWC (RF or BB data)
        2-D             NH.HW            NH (RF data).HW (RF data)
        1-D               H                      H (RF data)
            
        Args:
            inputs: Numpy array, real-valued or complex-type.
            hasN: Boolean, whether the first dimension represents amount N.
        Return:
            complex-valued array with the same dimension of inputs.
        
    Example:
        input = RF data, output = complex type BB data (a+bi)
        
        input = 2-channel real type BB data, output = complex type BB data (a+bi)
        
        input = complex type BB data, output = complex type BB data (a+bi)

    '''
    if inputs.dtype == np.complex64 or  inputs.dtype == np.complex128:
        return inputs
    shape = inputs.shape
    rank = inputs.ndim
    if rank not in {1,2,3,4}:
        raise ValueError(f'Unrecognized complex array with shape {shape}')
    # axes = {rank:axis for hilbert transform}
    axes = {
        1:-1, # [H]
        2:int(hasN), # NH or HW -> RF data
        3:int(hasN), # NHW or HW1 -> RF data, HW2 -> BB data
        4:1 # NHWC
        }
    if isrf(inputs, hasN):
        return Signal.hilbert(inputs, axis=axes[rank])
    else:
        real, imag = split_complex(inputs, hasN)
        return real + 1j*imag
    
    '''
    # the above codes equal to the below ones
    if rank == 1:
        return Signal.hilbert(inputs)
    if rank == 2:
        # [N,H] or [H,W] -> RF data
        return Signal.hilbert(inputs, axis=int(hasN))
    if rank == 3 and not hasN:
        # [H,W,C]
        if shape[-1]%2:
            # [H,W,1] -> RF data
            # real-valued signal needs hilbert transform
            return Signal.hilbert(inputs, axis=0)
        else:
            # [H,W,2] -> BB data in real-valued data format
            real, imag = split_complex(inputs)
            return real + 1j*imag
    if rank == 3 and hasN:
        # [N,H,W] -> RF data
        return Signal.hilbert(inputs, axis=1)
    if rank == 4 and hasN:
        # [N,H,W,C]
        if shape[-1]%2:
            # [N,H,W,1] -> RF data
            # real-valued signal needs hilbert transform
            return Signal.hilbert(inputs, axis=1)
        else:
            # [N,H,W,2] -> BB data in real-valued data format
            real, imag = split_complex(inputs)
            return real + 1j*imag
    '''
    
def convert_to_real(inputs, hasN=True):
    '''
    This function is used to convert data format from complex to 2-channel real array. 
    If input has been real data format, this function returns intput itself.
    
    Allow data shape includes:
                complex format (BB data)        real format
        4-D             NHWC                  NHWC (RF or BB data)
        3-D            NHW.HWC          NHW (RF data).HWC (RF or BB data)
        2-D             NH.HW            NH (RF data).HW (RF data)
        1-D               H                      H (RF data)
        
        Args:
            inputs: Numpy array, real-valued or complex-type.
            hasN: Boolean, whether the first dimension represents amount N.
        Return:
            real-valued array with the same dimension of inputs.
        
    Example:
        input = RF data, output = real type RF data
        
        input = complex-valued data, output = 2-channel real type BB data
        
        input = 2-channel real type BB data, output = 2-channel real type BB data
    '''
    if inputs.dtype != np.complex64 and  inputs.dtype != np.complex128:
        return inputs
    rank = inputs.ndim
    real, imag = split_complex(inputs, hasN)
    if rank == 4:
        # complex dtype [N,H,W,1] -> [N,H,W,2] real dtype
        return np.concatenate((real,imag),axis=-1)
    if rank in {1,2,3}:
        # complex dtype [H] -> [H,2] real dtype XXX not used XXX
        # complex dtype [H,W] -> [H,W,2] real dtype
        # complex dtype [N,H] -> [N,H,2] real dtype XXX not used XXX
        # complex dtype [N,H,W] -> [N,H,W,2] real dtype
        if rank == 3 and not hasN:
            # complex dtype [H,W,1] -> [H,W,2]
            return np.concatenate((real,imag),axis=-1)
        real = np.expand_dims(real, axis=-1)
        imag = np.expand_dims(imag, axis=-1)
        return np.concatenate((real,imag),axis=-1)
    raise ValueError(f'Unrecognized complex array with shape {inputs.shape}')

def _rf2bb_1data(signals, fc, fs):
    lpf = Signal.firwin(constant.FIRORDER, fc/(fs/2))
    t = np.arange(0,signals.shape[0])/fs
    if np.squeeze(signals).ndim == 1:
        bbsignal = signals*np.exp(-1j*2*np.pi*fc*t)
        return Signal.convolve(bbsignal, lpf, mode='same') # return 1D bb signal
    t = t.reshape((signals.shape[0],1)) # 2D time, shape [time, 1]
    bbsignal = reduce_dim(signals)*np.exp(-1j*2*np.pi*fc*t)
    bbsignal = np.apply_along_axis(Signal.convolve, 0, bbsignal, lpf, 'same')
    return bbsignal # return 2D bb signal

def rf2bb(signals, inds=None, fc=None, fs=None, return_format='imag'):
    '''
    Baseband demodulation. If the input "signals" is BB signal, it converts data format according to the "return_format".
    Args:
        signals: Numpy array, RF signal or BB signal.
        inds: int or Numpy array, indices of signals. If None, the default sampling rate and center frequency is based on the value in const file.
        retur_format: 'imag' or 'real', return 2-channel real format BB signal or complex format signal.
    Returns:
        Baseband signal in real or complex format.
    Allow data shape includes:
                complex format (BB data)        real format
        4-D             NHWC                  NHWC (RF or BB data)
        3-D            NHW.HWC          NHW (RF data).HWC (RF or BB data)
        2-D             NH.HW            NH (RF data).HW (RF data)
        1-D               H                      H (RF data)
    '''
    if return_format not in {'imag', 'real'}:
        raise ValueError(f"retrun_format is required to be either 'imag' or 'real' but get {return_format}")    
    shape = signals.shape # original shape
    if isinstance(inds, int):
        # BB data [1,H,W,1], [H,W,1], [H,W], [H] -> complex format [1,H,W,2], [H,W,2] -> real format
        # RF data [1,H,W,1], [1,H,W], [H,W,1], [1,H], [H,W], [H]
        N = 1 
        inds = [inds] 
        reduced_signals = np.squeeze(signals) # RF data [H,W], [H], BB data [H,W,2], [H,W], [H]
        reduced_signals = np.expand_dims(reduced_signals, axis=0) # RF data [1,H,W], [1,H], BB data [1,H,W,2], [1,H,W], [1,H]
    else:
        # BB data [N,H,W,1] -> complex format [N,H,W,2] -> real format
        # RF data [N,H,W,1], [N,H,W], [N,H]
        N = shape[0]
        reduced_signals = signals
    hasN = True
    if isbb(reduced_signals, hasN):
        # returns
        # BB data [1,H,W,1], [H,W,1], [H,W], [H] -> complex format [1,H,W,2], [H,W,2] -> real format
        # BB data [N,H,W,1] -> complex format [N,H,W,2] -> real format
        if return_format == 'imag':
            return convert_to_complex(signals)
        if return_format == 'real':
            return convert_to_real(signals)
    bbsignals = np.zeros(shape, dtype=np.complex64)
    # N = 1:
    #   [1,H,W], [1,H]
    # N != 1:
    #   [N,H,W,1], [N,H,W], [N,H]
    for ii in range(N):
        fc = fc if fc is not None else np.squeeze(get_data(inds[ii], 'f0'))
        fs = fs if fs is not None else get_sampling_rate(reduced_signals[ii], inds[ii], False)
        bbsignal = _rf2bb_1data(reduced_signals[ii], fc, fs) # [H,W] or [H]
        if N == 1:
            bbsignals = bbsignal
        else:
            if signals.ndim == 4:
                # [H,W] -> [H,W,1]
                bbsignals[ii] = np.expand_dims(bbsignal, axis=-1)
            else:
                bbsignals[ii] = bbsignal
        progressbar(ii+1, len(inds), 'Baseband demodulating')
    bbsignals = bbsignals.reshape(shape)
    if return_format == 'imag':
        return convert_to_complex(bbsignal, hasN).reshape(shape)
    else:
        return convert_to_real(bbsignals, hasN)

def _bb2rf_1data(signals, fc, fs):
    t = np.arange(0,signals.shape[0])/fs
    if np.squeeze(signals).ndim == 1:
        rfsignal = signals*np.exp(1j*2*np.pi*fc*t)
        return rfsignal.real # return 1D bb signal
    t = t.reshape((signals.shape[0],1)) # 2D time, shape [time, 1]
    rfsignal = reduce_dim(signals)*np.exp(1j*2*np.pi*fc*t)
    return rfsignal.real # return 2D bb signal

    
def bb2rf(signals, inds=None, fc=None, fs=None):
    '''
        signals: can be represented as complex-valued or two-channel real-valued data type. If it is 
        complex-valued data, shape can be [N,H,W], [N,H,W,1], or [H,W,1]. If it is two-channel real-valued
        data, shape can be [N,H,W,2] or [H,W,2].
    Allow data shape includes:
                complex format (BB data)        real format
        4-D             NHWC                  NHWC (RF or BB data)
        3-D            NHW.HWC          NHW (RF data).HWC (RF or BB data)
        2-D             NH.HW            NH (RF data).HW (RF data)
        1-D               H                      H (RF data)
        
    '''
    shape = signals.shape
    if isinstance(inds, int):
        # RF data -> [1,H,W,1], [1,H,W], [H,W,1], [1,H], [H,W], [H]
        # BB data -> [1,H,W,2], [H,W,2] and other complex format [1,H,W,1], [1,H,W], [H,W,1], [1,H], [H,W], [H]
        N = 1 
        inds = [inds]
        reduced_signals = np.squeeze(signals) # RF data [H,W], [H], BB data [H,W,2], [H,W], [H]
        reduced_signals = np.expand_dims(reduced_signals, axis=0) # RF data [1,H,W], [1,H], BB data [1,H,W,2], [1,H,W], [1,H]
    else:
        # BB data [N,H,W,1] -> complex format, [N,H,W,2] -> real format
        # BB data [N,H,W], [N,H] -> complex format
        N = shape[0]
        reduced_signals = signals
    hasN = True
    if isrf(reduced_signals, hasN):
        return signals
    else:
        # BB data [1,H,W,2] -> real format, [1,H,W], [1,H] -> complex format
        # BB data [N,H,W,2] -> real format, BB data [N,H,W,1] -> complex format 
        reduced_signals = convert_to_complex(reduced_signals) # convert to a+bi [N,H,W,2] -> [N,H,W,1], 
    rfsignals = np.zeros_like(reduced_signals, dtype=np.float32)
    # N = 1:
    #   [1,H,W,1], [1,H,W], [1,H]
    # N != 1:
    #   [N,H,W,1], [N,H,W], [N,H]
    for ii in range(N):
        fc = fc if fc is not None else np.squeeze(get_data(inds[ii], 'f0'))
        fs = fs if fs is not None else get_sampling_rate(reduced_signals[ii], inds[ii], False)
        rfsignal = _bb2rf_1data(reduced_signals[ii], fc, fs) # [H,W] or [H]
        if N == 1:
            rfsignals = rfsignal.astype(np.float32)
            return rfsignals.reshape(shape)
        else:
            if signals.ndim == 4:
                # [H,W] -> [H,W,1]
                rfsignals[ii] = np.expand_dims(rfsignal, axis=-1)
            else:
                rfsignals[ii] = rfsignal
        progressbar(ii+1, len(inds), 'Baseband demodulating')
    return rfsignals.astype(np.float32)

            
    
def bb2rf1D(signals, inds):
    '''
        signals: complex signals can be represented as complex-valued or two-channel real-valued data type. 
        If it is complex-valued data, shape can be [N,H], [N,H,1], or [H,1]. If it is two-channel real-valued
        data, shape can be [N,H,2] or [H,2].
    '''
    try:
        N = len(inds)
    except TypeError:
        N = len(np.asarray([inds]))
    
    if N == 1:
        # [H,1] or [H,2]
        if not np.iscomplex(signals).any():
            # real-valued data [H,2]
            signals = signals[:,:1] + 1j*signals[:,1:] # convert to complex-valued data [H,1]
        assert signals.shape[-1]%2 == 1 and signals.ndim == 2
        # complex-valued data [H,1]
        fc = np.squeeze(get_data(inds, 'f0'))
        fs = get_sampling_rate(signals, inds)
        t = np.arange(0,signals.shape[0])/fs
        t = t.reshape((signals.shape[0],1))
        RFsignal = signals*np.exp(1j*2*np.pi*fc*t)
        return RFsignal.real
    else:
        # [N,H], [N,H,1] or [N,H,2]
        if not np.iscomplex(signals).any():
            # real-valued data [N,H,2]
            signals = signals[:,:,:1] + 1j*signals[:,:,1:] # convert to complex-valued data [N,H,1]
        rank = signals.ndim
        if rank == 2:
            signals = np.expand_dims(signals, axis=2)
        RFsignals = np.zeros_like(signals, dtype=np.float32)
        for ii in range(len(signals)):
            fc = np.squeeze(get_data(inds[ii], 'f0'))
            fs = get_sampling_rate(signals[ii], inds[ii])
            t = np.arange(0,signals.shape[1])/fs
            t = t.reshape((-1,1)) # [H,1]
            RFsignal = signals[ii]*np.exp(1j*2*np.pi*fc*t) # [H,1]
            RFsignals[ii] = np.expand_dims(RFsignal.real, axis=0) # [N,H,1]
        if rank == 2:
            return np.squeeze(RFsignals)
        else:
            return RFsignals
    
def time2kspace(signals, hasN=True, shift=False):
    '''
    Transform signal from time space to frequency space, i.e. k-space.
    Allow data shape includes
        4-D: complex-valued NHWC (BB data) , real-valued NHWC (channel represents RF or BB data)
        3-D: complex-valued NHW (BB data), real-valued NHW (RF data), real-valued HWC (channel represents RF or BB data)
        2-D: complex-valued HW (BB data), real-valued HW (RF data)
    '''
    # axes = {rank:axis for fft2 i.e. the image shape of H,W }
    axes = {
        4: (1,2),
        3: (1,2) if hasN else (0,1),
        2: (0,1),
        }
    if isbb(signals):
        signals = convert_to_complex(signals, hasN)
    Signals = np.fft.fft2(signals, axes=axes[signals.ndim])
    if shift:
        Signals = np.fftshift(Signals)
    return convert_to_real(Signals)

def kspace2time(Signals, hasN=True, shift=False):
    '''
    Transform signal from frequency space, i.e. k-space, to time space.
    Allow data shape includes
        4-D: complex data with shape NHW1 (complex format) or NHW2 (real format)
        3-D: complex data with shape NHW (complex format) or HW2 (real format)
        2-D: complex data with shape HW (complex format)
    '''
    if shift:
        Signals = np.ifftshift(Signals)
    Signals = convert_to_complex(Signals)
    axes = {
        4: (1,2),
        3: (1,2) if hasN else (0,1),
        2: (0,1)
        }
    return np.fft.ifft2(Signals, axes=axes[Signals.ndim])     


def normalization(inputs, hasN=True):
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
    if rank not in {1,2,3,4}:
        raise ValueError(f"Unrecognized data type with shape {shape}")
    axes = {
        1:-1,
        2:1 if hasN else 0,
        3:(1,2) if hasN else None,
        4:(1,2,3)
        }
    if isbb(inputs):
        modulus = np.abs(inputs)
        return inputs/(np.max(modulus, axis=axes[rank], keepdims=True) + 1e-32)
    else:
        return inputs/(np.max(inputs, axis=axes[rank], keepdims=True) + 1e-32)
        
        
    # if rank == 2:
    #     # [H,W] in real or complex type
    #     return inputs/np.max(np.abs(inputs))
    # elif rank == 3:
    #     # only [H,W,C], [N,H,W] is NOT available
    #     if shape[-1]%2:
    #         # real type array
    #         return inputs/(np.max(np.abs(inputs)) + 1e-32)
    #     else:
    #         # complex type array
    #         real = inputs[:,:,:shape[-1]//2]
    #         imag = inputs[:,:,shape[-1]//2:]
    #         modulus = np.sqrt(real**2 + imag**2)
    #         return inputs/(np.max(modulus) + 1e-32)
    # elif rank == 4:
    #     if shape[-1]%2:
    #         # real type array, e.g. [N,H,W,1]
    #         return inputs/np.max(np.abs(inputs), axis=(1,2,3), keepdims=True)
    #     else:
    #         # complex type array, e.g. [N,H,W,2]
    #         real = inputs[:,:,:,:shape[-1]//2]
    #         imag = inputs[:,:,:,shape[-1]//2:]
    #         modulus = np.sqrt(real**2 + imag**2)
    #         return inputs/(np.max(modulus, axis=(1,2,3), keepdims=True) + 1e-32)
    # else:
    #     raise ValueError(f"Unrecognized data type with shape {shape}")
        
def standardization(inputs, hasN=True):
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
    if rank not in {1,2,3,4}:
        raise ValueError(f"Unrecognized data type with shape {shape}")
    axes = {
        1:-1,
        2:1 if hasN else 0,
        3:(1,2) if hasN else None,
        4:(1,2,3)
        }
    return (inputs - np.mean(inputs, axes[rank], keepdims=True))/(np.std(inputs, axes[rank], keepdims=True) + 1e-32)
    
    
    # def _standardization(x, axis=None):
    #     if axis is None:
    #         return (x - np.mean(x))/(np.std(x) + 1e-32)
    #     else:
    #         return (x - np.mean(x, axis, keepdims=True))/(np.std(x, axis, keepdims=True) + 1e-32)
    # if rank == 2:
    #     # [H,W] in real or complex type
    #     return _standardization(inputs)
    # elif rank == 3:
    #     # only [H,W,C], [N,H,W] is NOT available
    #     if shape[-1]%2:
    #         # real type array
    #         return _standardization(inputs)
    #     else:
    #         # complex type array
    #         real = inputs[:,:,:shape[-1]//2]
    #         imag = inputs[:,:,shape[-1]//2:]
    #         real = _standardization(real)
    #         imag = _standardization(imag)
    #         return np.concatenate((real,imag),axis=-1)
    # elif rank == 4:
    #     if shape[-1]%2:
    #         # real type array, e.g. [N,H,W,1]
    #         return _standardization(inputs, axis=(1,2,3))
    #     else:
    #         # complex type array, e.g. [N,H,W,2]
    #         real = inputs[:,:,:,:shape[-1]//2]
    #         imag = inputs[:,:,:,shape[-1]//2:]
    #         real = _standardization(real, axis=(1,2,3))
    #         imag = _standardization(imag, axis=(1,2,3))
    #         return np.concatenate((real, imag), axis=-1)
    # else:
    #     raise ValueError(f"Unrecognized data type with shape {shape}")
        
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
    
def focusing(img, ratio=0.05, hasN=True):
    '''
    In order to remove artifacts around edge. Support for data format NHWC, HWC, NHW, HW, NH, 1D.
        Args:
            img: Numpy array, displayed images.
            ratio: A decimal, clipping ratio of image.
        Return:
            Pruned numpy arrays.
    '''
    if ratio < 0 or ratio >= 1:
        raise ValueError('ratio is out of boundary')
    shape = img.shape
    if img.ndim == 4:
        # NHWC
        if not hasN:
            raise ValueError("The first dimension of a 4-D array must have N")
        H, W = round(shape[1]*ratio), round(shape[2]*ratio)
        return img[:,H:-H,W:-W,:]
    elif img.ndim == 3:
        if hasN: #NHW
            H, W = round(shape[1]*ratio), round(shape[2]*ratio)
            return img[:,H:-H,W:-W]
        else: # HWC
            H, W = round(shape[0]*ratio), round(shape[1]*ratio)
            return img[H:-H,W:-W,:]
    elif img.ndim == 2:
        if hasN: # NH
            H = round(shape[1]*ratio)
            return img[:,H:-H]
        else: # HW
            H, W = round(shape[0]*ratio), round(shape[1]*ratio)
            return img[H:-H,W:-W]
    elif img.ndim == 1:
        H = round(shape[0]*ratio)
        return img[H:-H]
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