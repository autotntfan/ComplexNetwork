# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 17:06:10 2022

@author: benzener
"""
import numpy as np
import os
if __name__ == '__main__':
    import sys
    currentpath = os.getcwd()
    addpath = os.path.dirname(os.path.dirname(currentpath))
    if addpath not in sys.path:
        sys.path.append(addpath)
    from baseband.utils.data_utils import angle, projection, envelope_detection, split_complex, normalization
    from baseband.utils.info import get_delaycurve
    from baseband.setting import constant
    sys.path.remove(addpath)
else:
    from .data_utils import angle, projection, envelope_detection, split_complex, normalization, focusing
    from .info import get_delaycurve
    from ..setting import constant
    
import tensorflow as tf
import pandas as pd
# ------------------------- Metrics -------------------------
def complex_diff(signal1, signal2, normalize=True):
    '''
    This function estimates the difference of real-part and imaginary-part
        Args:
            signal1: Numpy array.
            signal2: Numpy array, reference signal.
        Returns:
            Numpy array,
            real-part difference, imag-part difference
    '''
    if signal1.shape != signal2.shape:
        raise ValueError('Inputs are different size')
    if normalize:
        signal1 = normalization(signal1)
        signal2 = normalization(signal2)
    real1, imag1 = split_complex(signal1)
    real2, imag2 = split_complex(signal2)
    return np.abs(real1 - real2), np.abs(imag1 - imag2)

def phase_diff(signal1, signal2):
    '''
    This function estimates the difference of angle.
        Args:
            signal1: Numpy array.
            signal2: Numpy array, reference signal.
        Returns:
            Numpy array, angle difference
    '''
    if signal1.shape != signal2.shape:
        raise ValueError('Inputs are different size')
    return np.abs(angle(signal1) - angle(signal2))

def BPD(signal1, signal2, gain=0, *args, **kwargs):
    '''
    Beam pattern projection difference
    
    Args:
        signal1: ndarray with shape [N,H,W,C], [H,W], or [H,W,C]
        gain: scalar, gain for log compression.
        **kwargs:
            direction: string, only allow 'lateral' and 'axial'
            vmin: the minimum value of projection,
                i.e. vmin=0, gain=60 then value<0 would be forced to 0
                and the max value is 60
    Return:
        BPD score in scalar or vector.
    '''
    # beampattern projection difference
    assert signal1.shape == signal2.shape
    diff = np.abs(projection(signal1, gain, *args, **kwargs) - 
                  projection(signal2, gain, *args, **kwargs))
    if signal1.ndim == 4:
        try:
            assert diff.ndim == 2
        except AssertionError:
            diff = np.squeeze(diff, axis=-1)
        return np.mean(diff,axis=1)
    else:
        return np.mean(diff)
    
def IOU(signal1, signal2, DR=60, gain=0, gap=20):
    '''
    Calculate the intersection over union sets, i.e. IOU = A and B / A or B
    
    Args:
        signal1: Numpy array with shape [N,H,W,C] or [H,W,C]
        signal2: Numpy array with shape [N,H,W,C] or [H,W,C]
        gain: scalar, image displayed gain
        gap: scalar, 
        
    Returns:
        iou: ndarray, 
            M-by-N ndarray, where DR is divided into M intervals, e.g. DR=70, M=5, included
            ~0, 0~20, 20~40, 40~60, 60~70 dB.Each column contains M samples' IOU value during 
            specific range. Total N signals. Note that
            M = DR//gap + 1 if DR%gap is 0, otherwise M = DR//gap + 2 (i.e. 60~70 dB)
        DRs: list, dynamic range interval
        mask1: boolean ndarray with shape [M,N,H,W]. Binary mask of signal1 in each dynamic range interval
        mask2: boolean ndarray with shape [M,N,H,W]. Binary mask of signal2 in each dynamic range interval
        
    '''
    if signal1.shape != signal2.shape:
        raise ValueError('Inputs are different size')
    axis = (1,2) if signal1.ndim == 4 else None # summation along which axes
    DRs = [gaingap for gaingap in range(gain-DR, gain, gap)] + [gain] 
    signal1 = envelope_detection(signal1, gain)
    signal2 = envelope_detection(signal2, gain)
    mask1 = np.zeros((len(DRs),) + signal1.shape)
    mask2 = np.zeros((len(DRs),) + signal1.shape)
    for ii, DRrange in enumerate(DRs):
        if ii == 0:
            mask1[ii] = signal1 < DRrange
            mask2[ii] = signal2 < DRrange
            iou = np.sum(np.logical_and(signal1 < DRrange, signal2 < DRrange), axis=axis) / \
                np.sum(np.logical_or(signal1 < DRrange, signal2 < DRrange), axis=axis)
        else:
            R1 = np.logical_and(signal1 > DRs[ii-1], signal1 <= DRrange)
            R2 = np.logical_and(signal2 > DRs[ii-1], signal2 <= DRrange)
            mask1[ii] = R1
            mask2[ii] = R2
            iou = np.vstack([iou,np.sum(np.logical_and(R1, R2), axis=axis)/np.sum(np.logical_or(R1, R2), axis=axis)])
    return np.nan_to_num(iou,nan=0.0), DRs, mask1, mask2

def mse(signal1, signal2, focus=False, envelope=False, avg=True):
    kwargs = {
        'focus':focus,
        'envelope':envelope,
        'avg':avg
        }
    return __error_core(signal1, signal2, np.square, **kwargs)

def mae(signal1, signal2, focus=False, envelope=False, avg=True):
    kwargs = {
        'focus':focus,
        'envelope':envelope,
        'avg':avg
        }
    return __error_core(signal1, signal2, np.abs, **kwargs)

def ssim(signal1, signal2, focus=False, envelope=False, filter_size=7):
    kwargs = {
        'focus':focus,
        'envelope':envelope,
        'filter_size':filter_size
        }
    return __ssim_core(signal1, signal2, tf.image.ssim, **kwargs)

def ms_ssim(signal1, signal2, focus=False, envelope=False, filter_size=7):
    kwargs = {
        'focus':focus,
        'envelope':envelope,
        'filter_size':filter_size
        }
    return __ssim_core(signal1, signal2, tf.image.ssim_multiscale, **kwargs)

def save_metrics(signal1, signal2, model_name):
    save_dir = os.path.join(r'./modelinfo', model_name)
    file_name = os.path.join(save_dir, model_name + '_metrics.txt')
    focus = [False, False, True, True]
    envelope = [False, True, False, True]
    types = ['raw data', 'envelope', 'focus raw data', 'focus envelope']
    with open(file_name,'w') as f:
        for ii in range(4):
            f.write('\n' + types[ii] + ' metrics: \n')
            f.write('mse ' + str(mse(signal1, signal2, focus[ii], envelope[ii])) + '\n')
            f.write('mae ' + str(mae(signal1, signal2, focus[ii], envelope[ii])) + '\n')
            f.write('ssim ' + str(ssim(signal1, signal2, focus[ii], envelope[ii])) + '\n')
            f.write('ms_ssim ' + str(ms_ssim(signal1, signal2, focus[ii], envelope[ii])) + '\n')
            
def __preprocessing(signal1, signal2, focus=False, envelope=False):
    if focus:
        signal1 = focusing(signal1)
        signal2 = focusing(signal2)           
    if envelope:
        signal1 = envelope_detection(signal1)
        signal2 = envelope_detection(signal2)
    return signal1, signal2

def __error_core(signal1, signal2, func, **kwargs):
     assert signal1.shape == signal2.shape
     avg = kwargs['avg']
     kwargs.pop('avg')
     signal1, signal2 = __preprocessing(signal1, signal2, **kwargs)
     if signal1.ndim == 4 and not avg:
         return np.mean(np.sum(func(signal1 - signal2), axis=(1,2,3)))
     else:
         return np.mean(func(signal1 - signal2)) 
     
def __ssim_core(signal1, signal2, func, **kwargs):
    assert signal1.shape == signal2.shape
    filter_size = kwargs['filter_size']
    kwargs.pop('filter_size')
    signal1, signal2 = normalization(signal1), normalization(signal2)
    signal1, signal2 = __preprocessing(signal1, signal2, **kwargs)
    if kwargs['envelope']:
        return np.mean(func(signal1,
                            signal2,
                            max_val=1,
                            filter_size=filter_size).numpy())
    else:
        return np.mean(func(signal1,
                            signal2,
                            max_val=2,
                            filter_size=filter_size).numpy())

# ------------------------- statistics -------------------------
def err_statistic(signal1, signal2, levels, inds, *args, normalize=True, **kwargs):
    '''
    Calculate the difference of either or both real and imaginary part between signal1 and signal2.
    Args:
        signal1: ndarray with shape [N,H,W,C], [H,W], or [H,W,C], prediction.
        signal2: same data type as signal1, ground truth.
        levels: scalar or vector, phase aberration level.
        inds: scalar or vector, indices of signal1.
        **kwargs:
            gain: scalar, gain for log compression.
            vmin: the minimum value of projection,
                i.e. vmin=0, gain=60 then value<0 would be forced to 0
                and the max value is 60
    Returns
        err: 
            maxerr: [N,], max pixel-wise distance btw signal1 and signal2,
            sumerr: [N,], sum pixel-wise distance btw signal1 and signal2,
            LBPD: [N,], lateral beampattern projection difference.
            ABPD: [N,], axial beampatter projection difference.
            level: [N,], phase aberration levels.
            ind: [N,], indices of signal.
        err_2channel: error statistics for both real and imaginary part
        delay: [N, Nelements], delay curve for signal
    
    '''
    assert signal1.shape == signal2.shape
    if signal1.ndim == 4:
        N = signal1.shape[0]
    else:
        N = 1
    delay = np.zeros((N,constant.NELEMENT)) # delay curve
    for ii in range(N):
        delay[ii] = get_delaycurve(inds[ii])
    real_diff, imag_diff = complex_diff(signal1, signal2, normalize)
    err_2channel = {
        'level':np.tile(levels,2), # levels of real and imaginary part
        'maxerr':np.hstack([np.max(real_diff,axis=(1,2,3)),np.max(imag_diff,axis=(1,2,3))]), # max L1 norm of Re and Im channel for each signal
        'sumerr':np.hstack([np.sum(real_diff,axis=(1,2,3)),np.sum(imag_diff,axis=(1,2,3))]), # sum L1 norm of Re and Im channel for each signal
        'channel':np.asarray(['real']*(ii+1) + ['imag']*(ii+1)),
        'ind':np.tile(inds,2) # index of signal
        }
    err = {
        'maxerr':np.max(np.sqrt(real_diff**2 + imag_diff**2),axis=(1,2,3)), # max L2 norm for each signal
        'sumerr':np.sum(np.sqrt(real_diff**2 + real_diff**2),axis=(1,2,3)), # sum L2 norm for each signal
        'LBPD':BPD(signal1, signal2, direction='lateral', *args, **kwargs), 
        'ABPD':BPD(signal1, signal2, direction='axial', *args, **kwargs),
        'level':levels,
        'ind':inds
        }
    delay = {
        'delay':delay,
        'ind':inds
        }
    return err, err_2channel, delay

def iou_ratio(pred, ref, levels, threshold=0.5):
    if threshold > 1 or threshold < 0:
        raise ValueError(f"Threshold must be in the range of [0,1] but given {threshold}")
    # --------
    gain = 0
    DR = 60
    gap = 20
    # --------
    iou, DRs, _, _ = IOU(pred, ref, DR, gain, gap)
    iou_larger_than_half_count = np.zeros((len(DRs), 4))
    iou_larger_than_half_ratio = np.zeros((len(DRs), 5))
    for level in range(1,5):
        # phase aberration level
        level_n_iou = iou[:,levels==level]
        for iDR in range(iou.shape[0]):
            iou_larger_than_half_count[iDR,level-1] = len(list(filter(lambda x: x > threshold, level_n_iou[iDR])))
            iou_larger_than_half_ratio[iDR,level-1] = iou_larger_than_half_count[iDR,level-1]/len(level_n_iou[iDR])
    iou_larger_than_half_ratio[:,-1] = np.sum(iou_larger_than_half_count, axis=-1)/pred.shape[0]
    df = pd.DataFrame(np.round(iou_larger_than_half_ratio*100,2), 
                      columns=["level-1", "level-2", "level-3", "level-4", "Total"],
                      index=['I <= -60dB', '-60dB < I <= -40dB', '-40dB < I <= -20dB', '-20dB < I <= 0dB'])
    print(df)
    



    
