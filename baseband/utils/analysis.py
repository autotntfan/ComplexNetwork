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
    from baseband.utils import data_utils
    from baseband.utils.info import get_delaycurve, get_shape, isrf, isbb, check_format
    from baseband.setting import constant
    sys.path.remove(addpath)
else:
    from baseband.utils import data_utils
    from .info import get_delaycurve, get_shape, isrf, isbb, check_format
    from ..setting import constant
import cv2    
import tensorflow as tf
import pandas as pd
from scipy import signal

'''
allow dimension except for level-n figure:
    4-D NHWC
    3-D HWC
    2-D HW
allow dimension for level-n figure:
    4-D NHWC
'''

def _check(signal1, signal2=None, allow_ranks=None):
    if signal2 is not None:
        if signal1.shape != signal2.shape:
            raise ValueError(f'Inputs are different size of shape {signal1.shape} and {signal2.shape}')
        if signal1.ndim not in allow_ranks or signal2.ndim not in allow_ranks:
            raise ValueError("Input `signal` shape is not in allow dimensions")
    else:
        if signal1.ndim not in allow_ranks:
            raise ValueError("Input `signal` shape is not in allow dimensions")

# ------------------------- Metrics -------------------------
def complex_diff(signal1, signal2, return_ratio=False):
    '''
    This function computes the difference of real-part and imaginary-part.
    Allow shapes: NHWC, HWC, and HW
        Args:
            signal1: Numpy array.
            signal2: Numpy array, reference signal.
            return_ratio: Boolean, ratio = difference/2.
        Returns:
            Numpy array with the same shape of inputs
            real-part difference (ratio), imag-part difference (ratio)
    '''
    _check(signal1, signal2, {2,3,4})
    hasN = True if signal1.ndim == 4 else False
    if isrf(signal1):
        signal1 = data_utils.convert_to_complex(signal1)
    if isrf(signal2):
        signal2 = data_utils.convert_to_complex(signal2)
    signal1 = data_utils.normalization(signal1, hasN)
    signal2 = data_utils.normalization(signal2, hasN)
    real1, imag1 = data_utils.split_complex(signal1)
    real2, imag2 = data_utils.split_complex(signal2)
    real_diff = np.abs(real1 - real2)
    imag_diff = np.abs(imag1 - imag2)
    if return_ratio:
        return real_diff/2., imag_diff/2.
    else:
        return real_diff, imag_diff

def angle(signals):
    '''
    Compute the angle (phase) for complex value.
    Allow shapes: NHWC, HWC, and HW
        Args:
            signals: Numpy array, complex-valued or real-valued type.
        Return:
            Unwrapping angle
    '''
    _check(signals, allow_ranks={2,3,4})
    hasN = True if signals.ndim == 4 else False
    if isrf(signals):
        signals = data_utils.convert_to_complex(signals)
    complex_signal = check_format(signals, hasN, 'BB')
    wrapped_angle = np.angle(complex_signal)
    if wrapped_angle.ndim == 4:
        # [N,H,W,C]
        return np.apply_over_axes(np.unwrap, wrapped_angle, [1,2])
    else:
        # [H,W,C] or [H,W]
        return np.apply_over_axes(np.unwrap, wrapped_angle, [0,1])
    
def phase_diff(signal1, signal2):
    '''
    This function estimates the difference of angle.
        Args:
            signal1: Numpy array.
            signal2: Numpy array, reference signal.
        Returns:
            Numpy array, angle difference
    '''
    return np.abs(angle(signal1) - angle(signal2))

def BPD(signal1, signal2, direction='lateral', vmin=None, dBgain=None):
    '''
    Beam pattern projection difference
    
    Args:
        signal1: ndarray with shape [N,H,W,C], [H,W,C], or [H,W]
        signal2: ndarray with shape [N,H,W,C], [H,W,C], or [H,W]
        direction: string, only allow 'lateral' and 'axial'
        vmin: the minimum value of projection,
            i.e. vmin=0, gain=60 then value<0 would be forced to 0
            and the max value is 60
        dBgain: int, gain for log compression in log scale.
    Return:
        BPD score in scalar or vector.
    '''
    _check(signal1, signal2, {2,3,4})
    kwargs = {
        'direction':direction,
        'vmin':vmin,
        'dBgain':dBgain,
        'hasN':True if signal1.ndim == 4 else False
        }
    # beampattern projection difference
    diff = np.abs(data_utils.projection(signal1, **kwargs) - data_utils.projection(signal2, **kwargs))
    if signal1.ndim == 4:
        # [N,H,W,C]
        return np.mean(np.squeeze(diff), axis=-1)
    else: # [H,W,C] or [H,W]
        return np.mean(diff)
    
def IOU(signal1, signal2, DR=60, dBgain=0, gap=20):
    '''
    Calculate the intersection over union sets, i.e. IOU = A and B / A or B
    
    Args:
        signal1: Numpy array with shape [N,H,W,C], [H,W,C] or [H,W]
        signal2: Numpy array with shape [N,H,W,C], [H,W,C] or [H,W]
        DR: int, dynamic range in log scale
        gain: int, image displayed gain in log scale.
        gap: ing, intensity gap in log scale.
        
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
    _check(signal1, signal2, {2,3,4})
    axis = (1,2) if signal1.ndim == 4 else None # summation along which axes
    hasN = True if signal1.ndim == 4 else False
    DRs = [gaingap for gaingap in range(dBgain-DR, dBgain, gap)] + [dBgain] 
    signal1 = np.squeeze(data_utils.log_envelope(signal1, dBgain, hasN)) # [N,H,W] or [H,W]
    signal2 = np.squeeze(data_utils.log_envelope(signal2, dBgain, hasN))
    mask1 = np.zeros((len(DRs),) + signal1.shape) # [DRs, N, H, W]
    mask2 = np.zeros((len(DRs),) + signal1.shape)
    ious = np.zeros((len(DRs),get_shape(signal1, hasN, 'N'))) # [DRs, N]
    for ii, DRrange in enumerate(DRs):
        if ii == 0: # background
            R1 = signal1 < DRrange
            R2 = signal2 < DRrange
        else:
            R1 = np.logical_and(signal1 > DRs[ii-1], signal1 <= DRrange)
            R2 = np.logical_and(signal2 > DRs[ii-1], signal2 <= DRrange)
        mask1[ii] = R1
        mask2[ii] = R2
        ious[ii] = np.sum(np.logical_and(R1, R2), axis=axis)/np.sum(np.logical_or(R1, R2), axis=axis)
    return np.nan_to_num(ious,nan=0.0), DRs, mask1, mask2

def pulse_estimate(RFdata, Nc):
    N, H, W, C = RFdata.shape
    center_Aline = RFdata[:,:,W//2,:]
    hann_window = np.reshape(np.hanning(H), [1,-1,1])
    cepstrum = np.real(np.fft.ifft(np.log(np.abs(np.fft.fft(center_Aline*hann_window, axis=1))), axis=1));
    cepstrum[:,2:Nc+1,:] = 2*cepstrum[:,2:Nc+1,:]
    cepstrum[:,Nc+1:,:] = 0
    pulse = np.real(np.fft.ifft(np.exp(np.fft.fft(cepstrum, axis=1)), axis=1))
    pulse = pulse/np.max(np.abs(pulse), axis=(1,2), keepdims=True)
    return pulse

def mainlobe_pulse_diff(pred_psf, ref_psf, inds, return_aline=False):
    '''
    This function is used to calculate the central RF pulse difference btw prediction and reference.
    The central RF pulse is the central aline of the PSF where the mainlobe locates.
    Args:
        pred_psf: Numpy array with shape [N,H,W,C], [H,W,C] or [H,W], predicted baseband/RF PSF.
        ref_psf: Numpy array with shape [N,H,W,C], [H,W,C] or [H,W], reference baseband/RF PSF.
        inds: vector, indices of signal.
        return_aline: Boolean, whether to return the mainlobe pulse of prediction and reference.
    Return:
        pulse difference (and mainlobe pulse of prediction and reference of shape [N,H]).
    '''
    _check(pred_psf, ref_psf, {2,3,4})
    hasN = True if pred_psf.ndim == 4 else False
    if isbb(pred_psf) or isbb(ref_psf):
        pred_psf = data_utils.upsampling(data_utils.bb2rf(pred_psf, inds), 2, 'axial', hasN)
        ref_psf = data_utils.upsampling(data_utils.bb2rf(ref_psf, inds), 2, 'axial', hasN)
        
    W = get_shape(pred_psf, hasN, 'W')
    reduced_pred_psf = data_utils.precheck_dim(pred_psf, 3, hasN, True)
    reduced_ref_psf = data_utils.precheck_dim(ref_psf, 3, hasN, True) # force shape to [N,H,W]
    pred_mainlobe_aline = reduced_pred_psf[:,:,W//2]
    ref_mainlobe_aline = reduced_ref_psf[:,:,W//2]
    pred_mainlobe_aline = data_utils.apply_func(pred_mainlobe_aline, data_utils.normalization, np.squeeze) # [N,H]
    ref_mainlobe_aline = data_utils.apply_func(ref_mainlobe_aline, data_utils.normalization, np.squeeze) # [N,H]
    axis = 1 if hasN else 0
    pulse_diff = np.mean((pred_mainlobe_aline - ref_mainlobe_aline)**2, axis=axis) # [N,]
    if return_aline:
        return pulse_diff, pred_mainlobe_aline, ref_mainlobe_aline
    else:
        return pulse_diff  
         
def _error(signal1, signal2, func):
    _check(signal1, signal2, {2,3,4})
    hasN = True if signal1.ndim == 4 else False
    axes = {
        2:None,
        3:None,
        4:(1,2,3)
        }
    if isbb(signal1) or isbb(signal2):
        signal1, signal2 = data_utils.convert_to_complex(signal1, hasN), data_utils.convert_to_complex(signal2, hasN)
        diff = signal1 - signal2
        return np.mean(np.sum(func(diff.real) + func(diff.imag), axis=axes[signal1.ndim]))
    else:
        diff = signal1 - signal2
        return np.mean(np.sum(func(diff), axis=axes[signal1.ndim]))
    
def mae(signal1, signal2):
    return _error(signal1, signal2, np.abs)

def mse(signal1, signal2):
    return _error(signal1, signal2, np.square)

def _ssim_core(signal1, signal2, func, filter_size):
    _check(signal1, signal2, {2,3,4})
    hasN = True if signal1.ndim == 4 else False
    signal1 = data_utils.convert_to_real(data_utils.normalization(signal1, hasN), hasN)
    signal2 = data_utils.convert_to_real(data_utils.normalization(signal2, hasN), hasN)
    signal1 = data_utils.precheck_dim(signal1, 4, hasN, True)# convert to [N,H,W,C]
    signal2 = data_utils.precheck_dim(signal2, 4, hasN, True)
    if (signal1 < 0).any() or (signal2 < 0).any():
        # coherence signal
        signal1 = signal1 + 1
        signal2 = signal2 + 1
        max_val = 2
    else:
        # incoherence signal
        max_val = 1
    return np.mean(func(signal1,
                        signal2,
                        max_val=max_val,
                        filter_size=filter_size).numpy())
    
def ssim(signal1, signal2, filter_size=7):
    return _ssim_core(signal1, signal2, tf.image.ssim, filter_size)

def ms_ssim(signal1, signal2, filter_size=7):
    return _ssim_core(signal1, signal2, tf.image.ssim_multiscale,  filter_size)

def ssim_map(signal1, signal2, max_val=2, filter_size=15, filter_sigma=1.5, k1=0.01, k2=0.03):
    assert signal1.shape == signal2.shape
    kernelX = cv2.getGaussianKernel(filter_size, filter_sigma)
    window = kernelX * kernelX.T
    if signal1.ndim == 2: 
        H, W = signal1.shape
        C = 1
        N = 1
        signal1 = signal1.reshape(N,H,W,C)
        signal2 = signal2.reshape(N,H,W,C)
    elif signal1.ndim == 3:
        H, W, C = signal1.shape
        N = 1
        signal1 = signal1.reshape(N,H,W,C)
        signal2 = signal2.reshape(N,H,W,C)
    elif signal1.ndim == 4:
        N, H, W, C = signal1.shape
    if filter_size >= H or filter_size >= W:
        raise ValueError("filter size cannot be larger than width or height")
    ssim_map = np.zeros((N,H-filter_size+1,W-filter_size+1,C))
    C1 = (k1*max_val)**2
    C2 = (k2*max_val)**2
    for ithsignal in range(N):
        for ithchannel in range(C):
            mu1 = signal.convolve2d(signal1[ithsignal,:,:,ithchannel], window, 'valid')
            mu2 = signal.convolve2d(signal2[ithsignal,:,:,ithchannel], window, 'valid')
            mu1_sq = mu1*mu1
            mu2_sq = mu2*mu2
            mu1_mu2 = mu1*mu2 
            sigma1_sq = signal.convolve2d(signal1[ithsignal,:,:,ithchannel]*signal1[ithsignal,:,:,ithchannel], window, 'valid') - mu1_sq
            sigma2_sq = signal.convolve2d(signal2[ithsignal,:,:,ithchannel]*signal2[ithsignal,:,:,ithchannel], window, 'valid') - mu2_sq
            sigma12 = signal.convolve2d(signal1[ithsignal,:,:,ithchannel]*signal2[ithsignal,:,:,ithchannel], window, 'valid') - mu1_mu2   
            ssim_map[ithsignal,:,:,ithchannel] = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    mssim = np.mean(ssim_map)
    return mssim, np.squeeze(ssim_map)


def tf_ssim_map(signal1, signal2, max_val=2, filter_size=15, filter_sigma=1.5, k1=0.01, k2=0.03):
    assert signal1.shape == signal2.shape
    kernelX = cv2.getGaussianKernel(filter_size, filter_sigma)
    window = np.rot90(kernelX * kernelX.T,2)
    window = window.reshape(filter_size,filter_size,1,1)
    C1 = (k1*max_val)**2
    C2 = (k2*max_val)**2
    ithchannel = 0
    N,H,W,C = signal1.shape
    ssimmap = tf.zeros(N,H-filter_size+1,W-filter_size+1,C)
    while ithchannel < C:
        mu1 = tf.nn.conv2d(signal1[:,:,:,ithchannel:ithchannel+1], window, 1, 'VALID')
        mu2 = tf.nn.conv2d(signal2[:,:,:,ithchannel], window, 1, 'VALID')
        mu1_mu2 = tf.multiply(mu1,mu2)
        mu1_sq = tf.multiply(mu1,mu1)
        mu2_sq = tf.multiply(mu2,mu2)
        sigma1_sq = tf.subtract(tf.nn.conv2d(tf.multiply(signal1[:,:,:,ithchannel],signal1),window[:,:,:,ithchannel], 1, 'VALID'),mu1_sq)
        sigma2_sq = tf.subtract(tf.nn.conv2d(tf.multiply(signal2[:,:,:,ithchannel],signal2[:,:,:,ithchannel]),window, 1, 'VALID'),mu2_sq)
        sigma12 = tf.subtract(tf.nn.conv2d(tf.multiply(signal1[:,:,:,ithchannel],signal2[:,:,:,ithchannel]),window, 1, 'VALID'),mu1_mu2)
        ssimmap[:,:,:,ithchannel] = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        ithchannel = ithchannel + 1
    meanssim = tf.reduce_mean(ssimmap)
    return meanssim, ssimmap

def save_metrics(signal1, signal2, inds, levels, model_name):
    signal1, signal2 = data_utils.normalization(signal1), data_utils.normalization(signal2)
    _check(signal1, signal2, {4})
    save_dir = os.path.join(constant.MODELPATH, model_name)
    file_name = os.path.join(save_dir, model_name + '_metrics.txt')
    focus = [False, False, True, True]
    envelope = [False, True, False, True]
    def _generator(signal1, signal2, focus, envelope):
        if focus:
            if envelope:
                return data_utils.apply_func(signal1, data_utils.focusing, data_utils.envelope_detection),\
                    data_utils.apply_func(signal2, data_utils.focusing, data_utils.envelope_detection)
            else:
                return data_utils.focusing(signal1), data_utils.focusing(signal2)
        else:
            if envelope:
                return data_utils.apply_func(signal1, data_utils.envelope_detection),\
                    data_utils.apply_func(signal2, data_utils.envelope_detection)
            else:
                return signal1, signal2
    types = ['raw data', 'envelope', 'focus raw data', 'focus envelope']
    df_scores, df_ratio = leveln_IOU_metric(signal1, signal2, levels)
    with open(file_name,'w') as f:
        for ii in range(4):
            f.write('\n' + types[ii] + ' metrics: \n')
            f.write('mse ' + str(mse(*_generator(signal1, signal2, focus[ii], envelope[ii]))) + '\n')
            f.write('mae ' + str(mae(*_generator(signal1, signal2, focus[ii], envelope[ii]))) + '\n')
            f.write('ssim ' + str(ssim(*_generator(signal1, signal2, focus[ii], envelope[ii]))) + '\n')
            f.write('ms_ssim ' + str(ms_ssim(*_generator(signal1, signal2, focus[ii], envelope[ii]))) + '\n')
        f.write("\n Ratios of IOU larger than 0.5 \n" +  str(df_ratio) + "\n")
        f.write("\n IOU \n" + str(df_scores) + "\n")
        f.write("\n Beam pattern projection difference \n" + str(leveln_BPD_metric(signal1, signal2, levels)) + "\n")
        f.write("\n RF Mainlobe pulse difference \n" + str(leveln_MPD_metric(signal1, signal2, inds, levels)) + "\n")
            
# ------------------------- statistics -------------------------
def err_statistic(signal1, signal2, levels, inds, focus=True):
    '''
    Calculate the difference of either or both real and imaginary part between signal1 and signal2. Includes
    maximum absolute error and summation of absolute error for real and imaginary parts, and maximum squared error
    ,summation of squared error of complex signal and mainlobe difference of RF signal (BB is converted to RF).
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
            sumerr: [N,], sum pixel-wise distance btw signal1 and signal2
            LBPD: [N,], lateral beampattern projection difference.
            ABPD: [N,], axial beampatter projection difference.
            level: [N,], phase aberration levels.
            ind: [N,], indices of signal.
        err_2channel: error statistics for both real and imaginary part
        delay: [N, Nelements], delay curve for signal
    
    '''
    _check(signal1, signal2, {4})
    if focus:
        signal1, signal2 = data_utils.focusing(signal1), data_utils.focusing(signal2)
    N = get_shape(signal1, True, 'N')
    delay_curves = np.zeros((N,constant.NELEMENT), dtype=np.float32)
    for ii in range(N):
        delay_curves[ii] = np.squeeze(get_delaycurve(inds[ii]))
    real_diff, imag_diff = complex_diff(signal1, signal2)
    real_diff_ratio, imag_diff_ratio = complex_diff(signal1, signal2, True)
    err_2channel = {
        'Level':np.tile(levels,2), # levels of real and imaginary part
        'maxAE':np.hstack([np.max(real_diff,axis=(1,2,3)),np.max(imag_diff,axis=(1,2,3))]), # max L1 norm of Re and Im channel for each signal
        'MAE':np.hstack([np.mean(real_diff,axis=(1,2,3)),np.mean(imag_diff,axis=(1,2,3))]), # sum L1 norm of Re and Im channel for each signal
        'AER':np.hstack([np.max(real_diff_ratio, axis=(1,2,3)),np.max(imag_diff_ratio, axis=(1,2,3))]), # absoulte error ratio
        'Channel':np.asarray(['Real']*(ii+1) + ['Imag.']*(ii+1)),
        'ind':np.tile(inds,2) # index of signal
        }
    err = {
        'maxSE':np.max(np.sqrt(real_diff**2 + imag_diff**2),axis=(1,2,3)), # max L2 norm for each signal
        'MSE':np.mean(np.sqrt(real_diff**2 + real_diff**2),axis=(1,2,3)), # mean L2 norm (MSE) for each signal
        'LBPD':BPD(signal1, signal2, direction='lateral', dBgain=0), 
        'ABPD':BPD(signal1, signal2, direction='axial', dBgain=0),
        'MPD':mainlobe_pulse_diff(signal1, signal2, inds, False),
        'Level':levels,
        'ind':inds
        }
    return err, err_2channel

def _leveln_core(pred, ref, type_, levels, inds=None, focus=True):
    _check(pred, ref, {4})
    if type_ in {'MPD', 'MP'} and inds is None:
        raise ValueError("`levels`  and 'inds' cannot be None type")
    if focus:
        pred, ref = data_utils.focusing(pred), data_utils.focusing(ref)
    level_len = len(np.unique(levels))
    
    if type_ == 'IOU':
        iou, DRs, *_ = IOU(pred, ref)
        leveln_data = np.zeros((len(DRs), level_len), dtype=object)
        for iDR in range(iou.shape[0]):
            for level in range(1,level_len+1):
                leveln_data[iDR, level-1] = iou[iDR, levels==level]
        return leveln_data # shape of [DRs, levels, N']
    
    def _couple_leveln_data(data1, data2):
        leveln_data1 = np.zeros((level_len,), dtype=object)
        leveln_data2 = np.zeros((level_len,), dtype=object)
        for level in range(1,level_len+1):
            leveln_data1[level-1] = data1[levels==level]
            leveln_data2[level-1] = data2[levels==level]
        return leveln_data1, leveln_data2
    
    if type_ == 'cAER': # complex absoulte error ratio
        real_AER, imag_AER = complex_diff(pred, ref, True)
        return _couple_leveln_data(real_AER, imag_AER) # shape of [levels, N', H, W, C]

    if type_ == 'MP':
        _, pred_mainlobe_aline, ref_mainlobe_aline = mainlobe_pulse_diff(pred, ref, inds, True)
        return _couple_leveln_data(pred_mainlobe_aline, ref_mainlobe_aline) # shape of [levels, N', H]

    if type_ in {'LP', 'AP'}:
        direction = 'lateral' if type_ == 'LP' else 'axial'
        pred_proj = data_utils.projection(pred, direction, dBgain=0) 
        ref_proj = data_utils.projection(ref, direction, dBgain=0) 
        return _couple_leveln_data(pred_proj, ref_proj) # shape of [levels, N', W]
            
    leveln_data = np.zeros((level_len, ), dtype=object)
    if type_ == 'LBPD':
        data = BPD(pred, ref, direction='lateral', dBgain=0) # shape of [levels, N']
    elif type_ == 'ABPD':
        data = BPD(pred, ref, direction='axial', dBgain=0) # shape of [levels, N']
    elif type_ == 'MPD':
        data = mainlobe_pulse_diff(pred, ref, inds, False)  # shape of [levels, N']
    for level in range(1, level_len+1):
        leveln_data[level-1] = data[levels==level]
    return leveln_data

def leveln_IOU(pred, ref, levels, focus=True):
    return _leveln_core(pred, ref, 'IOU', levels, focus=focus)

def leveln_LBPD(pred, ref, levels, focus=True):
    return _leveln_core(pred, ref, 'LBPD', levels, focus=focus)

def leveln_ABPD(pred, ref, levels, focus=True):
    return _leveln_core(pred, ref, 'ABPD', levels, focus=focus)

def leveln_LP(pred, ref, levels, focus=True):
    return _leveln_core(pred, ref, 'LP', levels, focus=focus)

def leveln_AP(pred, ref, levels, focus=True):
    return _leveln_core(pred, ref, 'AP', levels, focus=focus)

def leveln_cAER(pred, ref, levels, focus=True):
    # complex absolute error ratio
    return _leveln_core(pred, ref, 'cAER', levels, focus=focus)

def leveln_MPD(pred, ref, levels, inds, focus=True):
    # main-lobe pulse difference
    return _leveln_core(pred, ref, 'MPD', levels, inds, focus)

def leveln_MP(pred, ref, levels, inds, focus=True):
    return _leveln_core(pred, ref, 'MP', levels, inds, focus)

def leveln_delay(levels, inds):
    level_len = len(np.unique(levels))
    leveln_data = np.zeros((level_len, ), dtype=object)
    data = np.array([get_delaycurve(ind) for ind in inds])
    for level in range(1, level_len+1):
        leveln_data[level-1] = data[levels==level]
    return leveln_data
    
def leveln_BPD_metric(pred, ref, levels, focus=True):
    '''
    Calculate LBPD and ABPD.
    Args:
        pred: ndarray with shape [N,H,W,C], prediction.
        ref: same data type as pred, ground truth.
        level: [N,], phase aberration levels.
        focus: 
    '''
    leveln_LBPDs, leveln_ABPDs = leveln_LBPD(pred, ref, levels, focus), leveln_ABPD(pred, ref, levels, focus)
    data_LBPD = []
    data_ABPD = []
    level_len = len(np.unique(levels))
    # mean ± std
    for level in range(level_len):
        data_LBPD.append(str(np.round(np.mean(leveln_LBPDs[level]),2)) + chr(177) + str(np.round(np.std(leveln_LBPDs[level]),2)))
        data_ABPD.append(str(np.round(np.mean(leveln_ABPDs[level]),2)) + chr(177) + str(np.round(np.std(leveln_ABPDs[level]),2)))
    column = ['level-'+str(ii+1) for ii in range(level_len)]
    df = pd.DataFrame(columns=column)
    df.loc['LBPD'] = data_LBPD
    df.loc['ABPD'] = data_ABPD
    return df

def leveln_IOU_metric(pred, ref, levels, threshold=0.5, focus=True):
    '''
    Calculate the ratio of iou larger than threshold.
    Args:
        pred: ndarray with shape [N,H,W,C], prediction.
        ref: same data type as pred, ground truth.
        level: [N,], phase aberration levels.
        threshold: float, iou threshold.
    '''
    if threshold > 1 or threshold < 0:
        raise ValueError(f"Threshold must be in the range of [0,1] but given {threshold}")
    leveln_iou = leveln_IOU(pred, ref, levels, focus)
    iou_larger_than_half_count = np.zeros_like(leveln_iou, dtype=np.float32) # the number of iou larger than threshold, shape = [DRs, levels]
    iou_larger_than_half_ratio = np.zeros_like(leveln_iou, dtype=np.float32) # the ratio of iou larger than threshold, shape = [DRs, levels+total]
    level_len = len(np.unique(levels))
    column = ['level-'+str(ii+1) for ii in range(level_len)]
    index = ['I <= -60dB', '-60dB < I <= -40dB', '-40dB < I <= -20dB', '-20dB < I <= 0dB']
    df_scores = pd.DataFrame(columns=column)
    for iDR in range(leveln_iou.shape[0]):
        data_iou = [] # iou value: mean ± std 
        for level in range(leveln_iou.shape[1]):
            # filter which x during the interval is larger than threshold
            iou_larger_than_half_count[iDR,level] = len(list(filter(lambda x: x > threshold, leveln_iou[iDR, level])))
            # convert to ratio
            iou_larger_than_half_ratio[iDR,level] = iou_larger_than_half_count[iDR,level]/len(leveln_iou[iDR, level])
            # mean ± std
            data_iou.append(str(np.round(np.mean(leveln_iou[iDR, level]),2)) + chr(177) + str(np.round(np.std(leveln_iou[iDR, level]),2)))
        df_scores.loc[index[iDR]] = data_iou    
    # calculate total ratio for each interval
    iou_larger_than_half_ratio = np.c_[iou_larger_than_half_ratio,np.sum(iou_larger_than_half_count, axis=-1)/pred.shape[0]]
    column.append('Total')
    df_ratio = pd.DataFrame(np.round(iou_larger_than_half_ratio*100,2), 
                            columns=column,
                            index=index)
    return df_scores, df_ratio

        
def leveln_MPD_metric(pred, ref, inds, levels, focus=True):
    '''
    Calculate the main-lobe pulse difference.
    Args:
        pred: ndarray with shape [N,H,W,C], prediction.
        ref: same data type as pred, ground truth.
        inds: [N,], indices of signals.
        level: [N,], phase aberration levels.
        threshold: float, iou threshold.
    '''
    level_len = len(np.unique(levels))
    leveln_pulse_diff = leveln_MPD(pred, ref, levels, inds, focus)
    data_pulse_diff = []
    for level in range(level_len):
        data_pulse_diff.append(str(np.round(np.mean(leveln_pulse_diff[level]),4)) + chr(177) + str(np.round(np.std(leveln_pulse_diff[level]),4)))
    df = pd.DataFrame(data_pulse_diff, index=['level-1','level-2','level-3','level-4'], columns=['MPD'])
    return df
        

    



    
