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
    from baseband.utils.data_utils import angle, projection, envelope_detection, split_complex, normalization, focusing, bb2rf1D
    from baseband.utils.info import get_delaycurve
    from baseband.setting import constant
    sys.path.remove(addpath)
else:
    from .data_utils import angle, projection, envelope_detection, split_complex, normalization, focusing, bb2rf1D
    from .info import get_delaycurve
    from ..setting import constant
import cv2    
import tensorflow as tf
import pandas as pd
from scipy import signal

# ------------------------- Metrics -------------------------
def complex_diff(signal1, signal2, normalize=True):
    '''
    This function estimates the difference of real-part and imaginary-part
        Args:
            signal1: Numpy array.
            signal2: Numpy array, reference signal.
            normalize: Boolean, whether to use normalization.
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

def mainlobe_pulse_diff(pred_bb_psf, ref_bb_psf, inds, return_aline=False):
    '''
    This function is used to calculate the central RF pulse difference btw prediction and reference.
    The central RF pulse is the central aline of the PSF where the mainlobe locates.
    Args:
        pred_bb_psf: Numpy array with shape [N,H,W,C], predicted baseband PSF.
        ref_bb_psf: Numpy array with shape [N,H,W,C], reference baseband PSF.
        inds: vector, indices of signal.
        return_aline: Boolean, whether to return the mainlobe pulse of prediction and reference.
    Return:
        pulse difference (and mainlobe pulse of prediction and reference).
    '''
    assert pred_bb_psf.shape == ref_bb_psf.shape
    N, H, W, C = pred_bb_psf.shape
    pred_bb_mainlobe_aline = pred_bb_psf[:,:,W//2,:]
    ref_bb_mainlobe_aline = ref_bb_psf[:,:,W//2,:]
    rf_pred_mainlobe_aline = np.squeeze(normalization(bb2rf1D(pred_bb_mainlobe_aline, inds)))
    rf_ref_mainlobe_aline = np.squeeze(normalization(bb2rf1D(ref_bb_mainlobe_aline, inds)))
    pulse_diff = np.sum((rf_pred_mainlobe_aline - rf_ref_mainlobe_aline)**2, axis=1)
    if return_aline:     
        return pulse_diff, rf_pred_mainlobe_aline, rf_ref_mainlobe_aline
    else:
        return pulse_diff
    
    

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
    return mssim,np.squeeze(ssim_map)


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
    save_dir = os.path.join(constant.MODELPATH, model_name)
    file_name = os.path.join(save_dir, model_name + '_metrics.txt')
    focus = [False, False, True, True]
    envelope = [False, True, False, True]
    types = ['raw data', 'envelope', 'focus raw data', 'focus envelope']
    df_scores, df_ratio = leveln_IOU_metric(signal1, signal2, levels)
    with open(file_name,'w') as f:
        for ii in range(4):
            f.write('\n' + types[ii] + ' metrics: \n')
            f.write('mse ' + str(mse(signal1, signal2, focus[ii], envelope[ii])) + '\n')
            f.write('mae ' + str(mae(signal1, signal2, focus[ii], envelope[ii])) + '\n')
            f.write('ssim ' + str(ssim(signal1, signal2, focus[ii], envelope[ii])) + '\n')
            f.write('ms_ssim ' + str(ms_ssim(signal1, signal2, focus[ii], envelope[ii])) + '\n')
        f.write("\n Ratios of IOU larger than 0.5 \n" +  str(df_ratio) + "\n")
        f.write("\n IOU \n" + str(df_scores) + "\n")
        f.write("\n Beam pattern projection difference \n" + str(leveln_BPD_metric(signal1, signal2, levels)) + "\n")
        f.write("\n RF Mainlobe pulse difference \n" + str(leveln_pulse_diff_metric(signal1, signal2, inds, levels)) + "\n")
            
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
        'Level':np.tile(levels,2), # levels of real and imaginary part
        'maxerr':np.hstack([np.max(real_diff,axis=(1,2,3)),np.max(imag_diff,axis=(1,2,3))]), # max L1 norm of Re and Im channel for each signal
        'sumerr':np.hstack([np.sum(real_diff,axis=(1,2,3)),np.sum(imag_diff,axis=(1,2,3))]), # sum L1 norm of Re and Im channel for each signal
        'Channel':np.asarray(['Real']*(ii+1) + ['Imag.']*(ii+1)),
        'ind':np.tile(inds,2) # index of signal
        }
    err = {
        'maxerr':np.max(np.sqrt(real_diff**2 + imag_diff**2),axis=(1,2,3)), # max L2 norm for each signal
        'sumerr':np.sum(np.sqrt(real_diff**2 + real_diff**2),axis=(1,2,3)), # sum L2 norm for each signal
        'LBPD':BPD(signal1, signal2, direction='lateral', *args, **kwargs), 
        'ABPD':BPD(signal1, signal2, direction='axial', *args, **kwargs),
        'pulsediff':mainlobe_pulse_diff(normalization(signal1), signal2, inds, False),
        'Level':levels,
        'ind':inds
        }
    delay = {
        'delay':delay,
        'ind':inds
        }
    return err, err_2channel, delay

def _leveln_core(pred, ref, levels, metric, focus=True):
    assert pred.shape == ref.shape
    if focus:
        pred, ref = focusing(pred), focusing(ref)
    level_len = len(np.unique(levels))
    leveln_data = np.zeros((level_len,), dtype=object)
    if metric == 'IOU':
        iou, DRs, *_ = IOU(pred, ref)
        leveln_data = np.zeros((len(DRs), level_len), dtype=object)
        for iDR in range(iou.shape[0]):
            for level in range(1,level_len+1):
                leveln_data[iDR,level-1] = iou[iDR, levels==level]
        return leveln_data
    elif metric == 'LBPD':
        data = BPD(pred, ref, direction='lateral')
    elif metric == 'ABPD':
        data = BPD(pred, ref, direction='axial')
    elif metric == 'LP':
        data = projection(pred, 0)
    elif metric == 'AP':
        data = projection(pred, 0, direction='axial')
    for level in range(1, level_len+1):
        leveln_data[level-1] = data[levels==level]
    return leveln_data

def leveln_IOU(pred, ref, levels, focus=True):
    return _leveln_core(pred, ref, levels, 'IOU', focus)

def leveln_LBPD(pred, ref, levels, focus=True):
    return _leveln_core(pred, ref, levels, 'LBPD', focus)

def leveln_ABPD(pred, ref, levels, focus=True):
    return _leveln_core(pred, ref, levels, 'ABPD', focus)

def leveln_LP(pred, ref, levels, focus=True):
    return _leveln_core(pred, ref, levels, 'LP', focus)

def leveln_AP(pred, ref, levels, focus=True):
    return _leveln_core(pred, ref, levels, 'AP', focus)
    
def leveln_BPD_metric(pred, ref, levels, **kwargs):
    leveln_LBPDs, leveln_ABPDs = leveln_LBPD(pred, ref, levels, **kwargs), leveln_ABPD(pred, ref, levels, **kwargs)
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

def leveln_IOU_metric(pred, ref, levels, threshold=0.5, **kwargs):
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
    leveln_iou = leveln_IOU(pred, ref, levels, **kwargs)
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

        
def leveln_pulse_diff_metric(pred, ref, inds, levels):
    pred = normalization(pred)
    ref = normalization(ref)
    pulse_diff = mainlobe_pulse_diff(pred, ref, inds)
    level_len = len(np.unique(levels))
    data_pulse_diff = []
    for level in range(1,level_len+1):
        leveln_pulse_diff = pulse_diff[levels==level]
        data_pulse_diff.append(str(np.round(np.mean(leveln_pulse_diff),2)) + chr(177) + str(np.round(np.std(leveln_pulse_diff),2)))
    column = ['level-'+str(ii+1) for ii in range(level_len)]
    df = pd.DataFrame(columns=column)
    df.loc['Pulse difference'] = data_pulse_diff
    return df
        

    



    
