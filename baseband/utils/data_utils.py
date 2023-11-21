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
    from baseband.utils.info import get_sampling_rate, get_data, progressbar, isrf, isbb, check_format
    sys.path.remove(addpath)
else:
    from ..setting import constant
    from .info import get_sampling_rate, get_data, progressbar, isrf, isbb, check_format
import scipy.signal      as Signal
import numpy as np
import cv2

def check_data_range(x, max_=1, min_=-1):
    '''
    Check the range of `x` is in [min_,max_].
    '''
    if np.max(x) > max_ or np.min(x) < min_:
        raise ValueError(f"input data is not in the range [{max_},{min_}], get [{np.min(x)},{np.max(x)}]")
        
def apply_func(x, *funcs):
    '''
    Apply multiple functions in `funcs` on x.
    '''
    for func in funcs:
        x = func(x)
    return x
     
       
def split_complex(x, hasN=True):
    '''
    Split complex value (2-channel real format or pure complex format) into real and imaginary part.
    If the "x" is RF data, x will be firstly convert to BB data and then splitted into real and imaginary part.
    Allow data shape includes:
                complex format (BB data)        real format
        4-D             NHWC                   NHWC (BB data)
        3-D            NHW.HWC                 HWC (BB data)
        2-D             NH.HW            
        1-D               H              
        
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
    if np.issubdtype(x.dtype, np.complexfloating):
        # return original shape
        return np.real(x), np.imag(x)
    shape = x.shape
    rank = x.ndim
    if rank not in {3,4}:
        raise ValueError(f'Unrecognized complex array with shape {shape}')
    if isrf(x, hasN):
        raise ValueError("Detect the inputs are RF signals")
    if rank == 3:
        return x[:,:,:shape[-1]//2], x[:,:,shape[-1]//2:]      
    if rank == 4:
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
    if np.issubdtype(inputs.dtype, np.complexfloating):
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
        print("Warning: Detect the `inputs` is RF data. It has been converted to baseband via Hilbert transform. ")
        return Signal.hilbert(inputs, axis=axes[rank])
    else:
        print("Warning: Detect the `inputs` are 2-branch complex signal in real format. It has been converted to be in complex format")
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
    if np.issubdtype(inputs.dtype, np.floating):
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

def precheck_dim(inputs, output_rank, input_hasN, output_hasN):
    '''
    Reshape dimension. If the rank of inputs is lower than the `output_rank`, this function expand the first dimension (N) when
    the `output_hasN` is True, otherwise, expand the last dimension. If the rank of inputs is larger than the `output_rank`, 
    this function squeeze the dimension and then expand the first dimension when the `output_hasN` is True, otherwise, 
    reuturn the squeezed data.
    
    Return data shape includes:
                complex format (BB data)        real format
        4-D             NHWC                  NHWC (RF or BB data)
        3-D            NHW.HWC          NHW (RF data).HWC (RF or BB data)
        2-D             NH.HW            NH (RF data).HW (RF data)
        1-D               H                      H (RF data)
    '''        
    input_rank = inputs.ndim
    output_shape = {
        4: np.array([1,1,1,1]),
        3: np.array([1,1,1,0]) if output_hasN else np.array([0,1,1,1]),
        2: [1,1,0,0] if output_hasN else np.array([0,1,1,0]),
        1: [0,1,0,0]
        }[output_rank]
    input_shape = {
        4: np.array([1,1,1,1]),
        3: np.array([1,1,1,0]) if input_hasN else np.array([0,1,1,1]),
        2: np.array([1,1,0,0]) if input_hasN else np.array([0,1,1,0]),
        1: np.array([0,1,0,0])
        }[input_rank]
    if input_rank < output_rank:
        if input_hasN:
            if not output_hasN:
                raise ValueError("Expanding dimension must preserve the channel dimension if it exists originally.")
        expand_axes = output_shape - input_shape
        if expand_axes[0]:
            inputs = np.expand_dims(inputs, 0)
        if expand_axes[2]:
            inputs = np.expand_dims(inputs,-1)
        if expand_axes[3]:
            inputs = np.expand_dims(inputs,-1)
    elif input_rank > output_rank:
        reduce_axes = input_shape - output_shape
        if reduce_axes[3]:
            inputs = np.squeeze(inputs, -1)
        if reduce_axes[2]:
            inputs = np.squeeze(inputs, -1)
        if reduce_axes[0]:
            inputs = np.squeeze(inputs, 0)
    else:
        if output_rank == 3:
            if output_hasN != input_hasN:
                inputs = np.transpose(inputs, (2,0,1))
            else:
                inputs = np.transpose(inputs, (1,2,0))
        elif output_rank == 2:
            if output_hasN != input_hasN:
                inputs = np.transpose(inputs, (1,0))
    assert inputs.ndim == output_rank
    return inputs

def _rf2bb_1data(signals, fc, fs):
    lpf = Signal.firwin(constant.FIRORDER, fc/(fs/2))
    t = np.arange(0,signals.shape[0])/fs
    if np.squeeze(signals).ndim == 1:
        bbsignal = signals*np.exp(-1j*2*np.pi*fc*t)
        return Signal.convolve(bbsignal, lpf, mode='same') # return 1D bb signal
    t = t.reshape((signals.shape[0],1)) # 2D time, shape [time, 1]
    bbsignal = precheck_dim(signals, 2, False, False)*np.exp(-1j*2*np.pi*fc*t)
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
    if isinstance(inds, int) or (np.issubdtype(inds.dtype, np.integer) and len(inds) == 1):
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
    reduced_signals = check_format(reduced_signals, hasN, 'rf')
    bbsignals = np.zeros(shape, dtype=np.complex64)
    # N = 1:
    #   [1,H,W], [1,H]
    # N != 1:
    #   [N,H,W,1], [N,H,W], [N,H]
    for ii in range(N):
        fc = fc if fc is not None else np.squeeze(get_data(inds[ii], 'f0'))
        fs = fs if fs is not None else get_sampling_rate(reduced_signals.shape[1], inds[ii], False)
        bbsignal = _rf2bb_1data(reduced_signals[ii], fc, fs) # [H,W] or [H]
        if N == 1:
            bbsignals = bbsignal
        else:
            if len(shape) == 4:
                # [H,W] -> [H,W,1]
                bbsignals[ii] = np.expand_dims(bbsignal, axis=-1)
            else:
                bbsignals[ii] = bbsignal
        progressbar(ii+1, len(inds), 'Baseband demodulating ...')
    if return_format == 'imag':
        return convert_to_complex(bbsignals, hasN).reshape(shape)
    else:
        shape = shape[:-1] + (2,)
        return convert_to_real(bbsignals, hasN).reshape(shape)

def _bb2rf_1data(signals, fc, fs):
    t = np.arange(0,signals.shape[0])/fs
    if np.squeeze(signals).ndim == 1:
        rfsignal = signals*np.exp(1j*2*np.pi*fc*t)
        return rfsignal.real # return 1D bb signal
    t = t.reshape((signals.shape[0],1)) # 2D time, shape [time, 1]
    rfsignal = precheck_dim(signals, 2, False, False)*np.exp(1j*2*np.pi*fc*t)
    return rfsignal.real # return 2D bb signal

    
def bb2rf(signals, inds=None, fc=None, fs=None):
    '''
        signals: can be represented as complex-valued or two-channel real-valued data type. 
    Allow data shape includes:
                complex format (BB data)        real format
        4-D             NHWC                  NHWC (RF or BB data)
        3-D            NHW.HWC          NHW (RF data).HWC (RF or BB data)
        2-D             NH.HW            NH (RF data).HW (RF data)
        1-D               H                      H (RF data)
        
    '''
    
    shape = signals.shape
    if isinstance(inds, int) or (np.issubdtype(inds.dtype, np.integer) and len(inds) == 1):
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
    reduced_signals = check_format(reduced_signals, hasN, 'bb') # convert to a+bi
    # N = 1:
    #   [1,H,W,1], [1,H,W], [1,H]
    # N != 1:
    #   [N,H,W,1], [N,H,W], [N,H] 
    rfsignals = np.zeros_like(reduced_signals, dtype=np.float32)
    for ii in range(N):
        fc = fc if fc is not None else np.squeeze(get_data(inds[ii], 'f0'))
        fs = fs if fs is not None else get_sampling_rate(reduced_signals.shape[1], inds[ii], False)
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
        progressbar(ii+1, len(inds), 'Modulating ...')
    return rfsignals.astype(np.float32)

def time2kspace(signals, hasN=True, shift=False):
    '''
    Transform signal from time space to frequency space, i.e. k-space.
    Allow data shape includes:
                complex format (BB data)        real format
        4-D             NHWC                  NHWC (RF or BB data)
        3-D            NHW.HWC          NHW (RF data).HWC (RF or BB data)
        2-D             NH.HW            NH (RF data).HW (RF data)
        1-D               H                      H (RF data)
    Return:
        Complex single format k-space signal
    '''
    # axes = {rank:axis for fft2 i.e. the image shape of H,W }
    axes = {
        4: (1,2),
        3: (1,2) if hasN else (0,1),
        2: -1 if hasN else (0,1),
        }
    if isbb(signals):
        signals = convert_to_complex(signals, hasN)
    rank = signals.ndim
    if rank == 1:
        Signals = np.fft.fft(signals)
    elif rank == 2 and hasN:
        Signals = np.fft.fft(signals, axis=axes[rank])
    else:
        Signals = np.fft.fft2(signals, axes=axes[rank])
    if shift:
        Signals = np.fft.fftshift(Signals)
    return Signals.astype(np.complex64)

def kspace2time(Signals, hasN=True, shift=False):
    '''
    Transform signal from frequency space, i.e. k-space, to time space.
    Allow data shape includes:
                complex format                     real format
        4-D             NHWC                  NHWC (real or complex data)
        3-D            NHW.HWC          NHW (real data).HWC (real or complex data)
        2-D             NH.HW            NH (real data).HW (real data)
        1-D               H                      H (real data)
    
    Args:
        Signals: ndarray, k-space data in complex format or 2-branch complex data in real format.
        hasN: Boolean, whether the first dimension represents amount N.
    Return:
        Complex single format time-space signal
        
    '''
    if shift:
        Signals = np.fft.ifftshift(Signals)
    Signals = convert_to_complex(Signals, hasN)
    axes = {
        4: (1,2),
        3: (1,2) if hasN else (0,1),
        2: (0,1)
        }
    signals = np.fft.ifft2(Signals, axes=axes[Signals.ndim])
    if np.issubdtype(signals.dtype, np.floating):
        return signals.astype(np.float32)
    elif np.issubdtype(signals.dtype, np.complexfloating):
        return signals.astype(np.complex64)


def normalization(inputs, hasN=True):
    '''
    Limit input in the range of [-1,1] for coherent signal, or [0,1]
    for incoherent signal. If the `inputs` is 2-branch complex data in real format,
    it automatically returns the normalized complex data in complex format.
    
    Allow data shape includes:
                complex format (BB data)        real format
        4-D             NHWC                  NHWC (RF or BB data)
        3-D            NHW.HWC          NHW (RF data).HWC (RF or BB data)
        2-D             NH.HW            NH (RF data).HW (RF data)
        1-D               H                      H (RF data)
    Args:
        inputs: ndarray
        hasN: Boolean, whether the first dimension represents amount N.
    
    Return:
        ndarray, max value = orignal value/abs(max value) with the sample shape of inputs
    
    '''
    shape = inputs.shape
    rank = inputs.ndim
    if rank not in {1,2,3,4}:
        raise ValueError(f"Unrecognized data type with shape {shape}")
    axes = {
        1:-1,
        2:1 if hasN else None,
        3:(1,2) if hasN else None,
        4:(1,2,3)
        }
    if isbb(inputs):
        inputs = convert_to_complex(inputs, hasN)
        modulus = np.abs(inputs)
        outputs = inputs/(np.max(modulus, axis=axes[rank], keepdims=True) + constant.EPS)
    else:
        outputs = inputs/(np.max(np.abs(inputs), axis=axes[rank], keepdims=True) + constant.EPS)
    return outputs
        
        
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
    Allow data shape includes:
                complex format (BB data)        real format
        4-D             NHWC                  NHWC (RF or BB data)
        3-D            NHW.HWC          NHW (RF data).HWC (RF or BB data)
        2-D             NH.HW            NH (RF data).HW (RF data)
        1-D               H                      H (RF data)
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
        2:1 if hasN else None,
        3:(1,2) if hasN else None,
        4:(1,2,3)
        }
    if isbb(inputs):
        inputs = convert_to_complex(inputs, hasN)
    return (inputs - np.mean(inputs, axes[rank], keepdims=True))/(np.std(inputs, axes[rank], keepdims=True) + constant.EPS)

def envelope_detection(signal, hasN=True):
    '''
    Detect envelope, where gain is equivalent to dynamic range. However, actually
    gain is not equal to DR always. It should be envelope_detection(self, signal, DR, gain).
    Allow shape:
                complex format (BB data)        real format
        4-D             NHWC                  NHWC (RF or BB data)
        3-D            NHW.HWC          NHW (RF data).HWC (RF or BB data)
        2-D             NH.HW            NH (RF data).HW (RF data)
        1-D               H                      H (RF data)
        Args:
            signal: Numpy array, it could be real or complex type.
            
        Return:
            Envelope data in linear scale of the same dimension.
    
    '''
    signal = normalization(signal, hasN)
    envelope = np.abs(convert_to_complex(signal, hasN))
    return envelope
    
        
def log_compression(signal, dBgain=0, hasN=True):
    '''
    Convert linear scale to log scale. The `dBgain` is the gain compensation.
    Allow shape:
                complex format (BB data)        real format
        4-D             NHWC                  NHWC (RF or BB data)
        3-D            NHW.HWC          NHW (RF data).HWC (RF or BB data)
        2-D             NH.HW            NH (RF data).HW (RF data)
        1-D               H                      H (RF data)
    *The log(0) is undefined, hence that is added by EPS, i.e. at least -20*log(EPS) dB.
    Args:
        signal: Numpy array, it could be real or complex type.
        dBgain: scalar, gain for log compression.
    Return:
        Numpy array in log scale with `dBgain` gain of the same dimension.
    '''
    if signal.ndim == 4:
        # [N,H,W,1]
        return 20*np.log10(signal/np.max(signal,axis=(1,2,3),keepdims=True) + constant.EPS) + dBgain
    elif signal.ndim == 3:
        if hasN:
            # [N,H,W]
            return 20*np.log10(signal/np.max(signal, axis=(1,2), keepdims=True) + constant.EPS) + dBgain
        else:
            # [H,W,1]
            return 20*np.log10(signal/np.max(signal) + constant.EPS) + dBgain
    elif signal.ndim == 2:
        if hasN:
            # [N,H]
            return 20*np.log10(signal/np.max(signal, axis=1, keepdims=True) + constant.EPS) + dBgain
        else:
            # [H,W]
            return 20*np.log10(signal/np.max(signal) + constant.EPS) + dBgain
    elif signal.ndim == 1:
        return 20*np.log10(signal/np.max(signal) + constant.EPS) + dBgain
    else:
        raise ValueError('Too many dimension of the input `signal`.')
        
def log_envelope(signal, dBgain=0, hasN=True):
    '''
    This function combines the envelope detection and log compression.
    Args:
        signal: Numpy array, it could be real or complex type.
        dBgain: scalar, gain for log compression.
    Return:
        log scale envelope data.
    '''
    envelope = envelope_detection(signal, hasN)
    return log_compression(envelope, dBgain, hasN)
    

    
def projection(signal, direction='lateral', vmin=None, dBgain=None, hasN=True):
    '''
    Axial or lateral projection of signal Only support for RF data.
    Allow shape:
                 real and complex format
        4-D             NHWC                 
        3-D            NHW,HWC          
        2-D              HW                
        
    Args:
        signal: ndarray, target
        gain: scalar, gain for log compression.
        direction: string, only allow 'lateral' and 'axial'
        vmin: the minimum value of projection,
            i.e. vmin=0, gain=60 then value<0 would be forced to 0
            and the max value is 60
        hasN: Boolean, whether the first dimension represents amount N.
    Return:
        1-D/2-D projection data along axial or lateral determined
        by input sequence. Output shape is either NH or NW.
    '''
    if direction not in {'lateral','axial'}:
        raise ValueError("The `direction` should be either 'lateral' or 'axial'.")
    if signal.ndim == 4:
        # [N,H,W,C]
        axis = 1 if direction == 'lateral' else 2
    elif signal.ndim == 3:
        # [N,H,W] or [H,W,C]
        if hasN:
            axis = 1 if direction == 'lateral' else 2
        else:
            axis = 0 if direction == 'lateral' else 1
    elif signal.ndim == 2:
        axis = 0 if direction == 'lateral' else 1
    else:
        raise ValueError(f'Unsupport dimension {signal.ndim}')
    if dBgain is None:
        signal = check_format(signal, hasN, 'rf')
        # directly applied projection without log compression
        return np.squeeze(np.max(signal, axis, initial=vmin))
    else:
        # do log compression depending on DR
        outputs = np.max(log_envelope(signal, dBgain, hasN), axis, initial=vmin)
        return np.squeeze(outputs)
    

    
def focusing(img, ratio=0.05, hasN=True):
    '''
    In order to remove artifacts around edge. Support for data format NHWC, HWC, NHW, HW, NH, 1D.
        Args:
            img: Numpy array, displayed images.
            ratio: A decimal, clipping ratio of image.
            hasN: Boolean, whether the first dimension represents amount N.
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
        
def downsampling(signal, factor, direction='lateral', hasN=True):
    '''
    Downsample signal by a factor along a specific direction.
    Allow shapes:
        complex format (BB data)              real format
    4-D             NHWC                  NHWC (RF or BB data)
    3-D            NHW.HWC          NHW (RF data).HWC (RF or BB data)
    2-D             NH.HW            NH (RF data).HW (RF data)
    1-D               H                      H (RF data)
    
        Args:
            signal: ndarray.
            factor: int, decimation factor.
            direction: string, it can be 'lateral', 'axial', or 'both'
            hasN: Boolean, whether the first dimension represents amount N.
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
        if hasN:
            # [N,H,W]
            if direction == 'axial':
                return signal[:,::factor,:]
            elif direction == 'lateral':
                return signal[:,:,::factor]
            else:
                return signal[:,::factor,::factor]
        else:
            # [H,W,C]
            if direction == 'lateral':
                return signal[:,::factor,:]
            elif direction == 'axial':
                return signal[::factor,:,:]
            else:
                return signal[::factor,::factor,:]
    elif signal.ndim == 2:
        if hasN:
            # [N,H]
            return signal[:,::factor]
        else:
            # [H,W]
            if direction == 'lateral':
                return signal[:,::factor]
            elif direction == 'axial':
                return signal[::factor,:]
            else:
                return signal[::factor,::factor]
    elif signal.ndim == 1:
        return signal[::factor]
    else:
        raise ValueError('Only support 2D,3D,and 4D signal')
        
def upsampling(signal, factor, direction='lateral', hasN=True):
    '''
    Upsample signal by a factor along a specific direction. If signal is 2D image, it upsample the signal via
    bilinear interpolation (See tensorflow.image.imresize). If signal is 1D, it upsample via polyphase filtering.
 
    Allow shape:
                    real format
        4-D             NHWC                 
        3-D            NHW,HWC          
        2-D             NH,HW    
        1-D               H

        Args:
            signal: ndarray with shape [N,W,H,C], [H,W,C], [N,H,W], or [H,W].
            factor: int, decimation factor.
            direction: string, it can be 'lateral', 'axial', or 'both'
        Return:
            ndarray
        e.g. input size = [10,257,257,2], factor = 2, direction='both'
            output size = [10,129,129,2]
            
    '''
    import tensorflow as tf
    from scipy.signal import resample_poly
    if isbb(signal):
        raise ValueError("Input `signal` can only be real-valued signal.")
    if direction not in {'lateral','axial','both'}:
        raise ValueError('Direction only allows lateral, axial, and both')
    if signal.ndim == 4:
        # [N,H,W,C]
        _, H, W, _ = signal.shape
        if direction == 'lateral':
            W = factor*W
        elif direction == 'axial':
            H = factor*H
        else:
            H = factor*H
            W = factor*W
        return tf.image.resize(signal, [H,W]).numpy()
    elif signal.ndim == 3:
        if hasN:
            _, H, W = signal.shape
        else:
            H, W, _ = signal.shape
        if direction == 'lateral':
            W = factor*W
        elif direction == 'axial':
            H = factor*H
        else:
            H = factor*H
            W = factor*W
        return tf.image.resize(signal, [H,W]).numpy()
    elif signal.ndim == 2:
        if hasN:
            # [N,H]
            return np.apply_along_axis(resample_poly, 1, signal, factor, 1, 1)
        else:
            # [H,W]
            H, W = signal.shape
            signal = np.expand_dims(signal, -1)
            return np.squeeze(tf.image.resize(signal, [H,W]).numpy())
    elif signal.ndim == 1:
        return resample_poly(signal, factor, 1)
    else:
        raise ValueError('Too many dimension of the input `signal`.')

        
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


    