# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 22:10:57 2022

@author: benzener
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
if __name__ == '__main__':
    import sys
    currentpath = os.getcwd()
    addpath = os.path.dirname(os.path.dirname(currentpath))
    if addpath not in sys.path:
        sys.path.append(addpath)
    from baseband.setting import constant
    from baseband.utils import data_utils, info, analysis
    sys.path.remove(addpath)
else:
    from ..setting import constant
    from baseband.utils import data_utils, info, analysis

'''
Allow dimension of drawing figure:
    NHWC(N=1), HWC, HW
'''


def _check(img):
    if img.ndim not in {2,3,4}:
        raise ValueError("Input image shape should be NHWC, HWC, or HW.")
    hasN = True if img.ndim == 4 else False
    img = data_utils.precheck_dim(img, 2, hasN, False) # reshape to [H,W]
    hasN = False
    return img, hasN
            
# ------------------------- basic figure -------------------------
def save_fig(fig, saved_name):
    '''
    save figure to specific path
    
    Args:
        String
        model_name: model path
        saved_name: saved figure name
        saved_dir: desired directory will be built if it does not exist
        fig: saved figure
        
    '''
    saved_dir = os.path.abspath(os.path.join(saved_name, os.pardir))
    if not os.path.exists(saved_dir):
        # if directory doesn't exist then built
        try:
            os.mkdir(saved_dir)
        except FileNotFoundError:
            os.makedirs(saved_dir) # make parent directory
    if fig:
        fig.savefig(saved_name, dpi=300)
    else:
        plt.savefig(saved_name, dpi=300)
        
def gray(img, DR=60, dBgain=0, axis=None, title_name=None, show=True, saved_name=None):
    '''
    Show grayscale image. The default displayed range of color bar under 0. e.g. color range [-60,0] 
    whose dynamic range is 60 and gain is 0. Otherwise, the displayed range of color bar is [gain-DR, gain].
        Args:
            img: A numpy array, displayed image has only ONE color channel for grayscale of shape []
            DR: scalar, dynamic range
            dBgain: scalar, image displayed gain
            axis: A tuple, displayed axis consists of (xmin, xmax, ymax, ymin) for image
            title_name: string, title of plot
            kwargs:
                model_name: string, saving directory of which model
                saved_name: string, suffix of saved image. e.g. model_name/model_name+suffix.png
                saved_dir: desired directory will be built if it does not exist
        return:
            ONLY display gray-scale image
            
    '''
    img, _ = _check(img)
    fig = plt.figure()
    plt.imshow(img, cmap='gray', vmin=dBgain-DR, vmax=dBgain, extent=axis, aspect='auto')
    if axis is not None:
        plt.xlabel('Lateral position (mm)')
        plt.ylabel('Depth (mm)')
    if title_name is not None:
        plt.title(title_name)
    plt.colorbar()
    if saved_name is not None:
        save_fig(fig, saved_name)
    plt.show() if show else plt.close('all')


def heatmap(img, axis=None, title_name=None, show=True, saved_name=None, **kwargs):
    '''
    Show heatmap image.
        Args:
            img: A numpy array, displayed image has only ONE color channel for grayscale.
            axis: A tuple, displayed axis consists of (xmin, xmax, ymax, ymin) for image
            title_name: string, title of plot
            kwargs:
                model_name: string, saving directory of which model
                saved_name: string, suffix of saved image. e.g. model_name/model_name+suffix.png
                saved_dir: desired directory will be built if it does not exist
        return:
            ONLY display heatmap image
            
    '''
    img, _ = _check(img)
    fig = plt.figure()
    if axis is not None:
        xlabel = np.linspace(axis[0],axis[1],img.shape[1])
        ylabel = np.linspace(axis[3],axis[2],img.shape[0])
        sns.heatmap(img, cmap="RdBu_r", xticklabels=xlabel, yticklabels=ylabel, **kwargs)
        plt.xlabel('Lateral position (mm)')
        plt.ylabel('Depth (mm)')
    else:
        sns.heatmap(img, cmap="RdBu_r", **kwargs)
    if title_name is not None:
        plt.title(title_name)
    if saved_name is not None:
        save_fig(fig, saved_name)
    plt.show() if show else plt.close('all')


def boxplot(data, xlabel, ylabel, hue=None, title_name=None, show=True, saved_name=None):
    '''
    Show box plot.
        Args:
            data: 
            xlabel
            ylabel
            hue
            title_name: string, title of plot
            kwargs:
                model_name: string, saving directory of which model
                saved_name: string, suffix of saved image. e.g. model_name/model_name+suffix.png
                saved_dir: desired directory will be built if it does not exist
        return:
            ONLY display box plot
    '''
    fig = plt.figure()
    sns.boxplot(data=data, x=xlabel, y=ylabel, hue=hue)
    if title_name is not None:
        plt.title(title_name)
    if saved_name is not None:
        save_fig(fig, saved_name)
    plt.show() if show else plt.close('all')


def Bmode_fig(img, DR=60, dBgain=0, axis=None, title_name='Bmode', show=True, saved_name=None):
    '''
    Show the B-mode image in grayscale.
        Args:
            img: Numpy array, displayed image.
            DR: scalar, dynamic range.
            gain: scalar, image display gain.
            title_name: String, figure title
            ind: An integer, the index of displayed image. In order to get displayed axis.
            saved_name: String, saved name.
            kwargs:
                model_name: string, saving directory of which model
                saved_dir: desired directory will be built if it does not exist
    '''
    if img.ndim not in {2,3,4}:
        raise ValueError("Input image shape should be NHWC, HWC, or HW.")
    hasN = True if img.ndim == 4 else False    
    img = data_utils.log_envelope(img, dBgain, hasN)
    img = data_utils.precheck_dim(img, 2, hasN, False) # reshape to [H,W]
    gray(img, DR, dBgain, axis, title_name, show, saved_name)      
    
    
def fft_fig(signal, ind=None, fs=None, Aline=False, title_name=None, show=True, saved_name='Spectrum'):
    '''
    Show FFT. Allow shape NHWC, HWC, HW, H
        Args:
            img: Numpy array, analyzed image.
            ind: An integer, the index of displayed image. In order to get sampling rate.
            Aline: Boolean, only compute the center signal.
    '''
    hasN = True if signal.ndim == 4 else False
    if signal.ndim > 1:
        signal = data_utils.precheck_dim(signal, 2, hasN, False)
    if fs is None:
        fs = info.get_sampling_rate(signal.shape[0], ind)
    Signal = data_utils.time2kspace(signal, False, shift=True)
    if Aline and signal.ndim > 1:
        # check the spectrum of the central Aline
        center = signal.shape[1]//2
        Signal = np.abs(data_utils.time2kspace(signal[:,center], False, shift=True)) # 1D FFT -> output shape = [H]
    else:
        Signal = np.abs(data_utils.time2kspace(signal, False, shift=True)) # 2D FFT for [H,W] or 1D FFT for [H]
    freq_axis = np.linspace(-fs/2, fs/2, Signal.shape[0])/1e6
    fc = freq_axis[np.argmax(Signal)]
    fig = plt.figure()
    plt.plot(freq_axis, Signal)
    plt.xlabel('MHz')
    plt.ylabel('Magnitude')
    plt.xlim([-10,10])
    if title_name is not None:
        plt.title(title_name)
    else:
        plt.title(f'Spectrum fc = {fc}')
    if saved_name is not None:
        save_fig(fig, saved_name)
    plt.show() if show else plt.close('all')

    
    
def angle_fig(signal, axis=None, title_name=None, show=True, saved_name=None):
    '''
    Show phase distribution.
    '''
    signal, hasN = _check(signal)
    ang = analysis.angle(signal)
    heatmap(ang.astype(np.float32), axis, title_name, show, saved_name)

   
def complex_distribution_fig(signal, axis=None, title_name=None, show=True, saved_name=None):
    '''
    Show real- and imaginary-part distribution.
    '''
    signal, hasN = _check(signal)
    signal = data_utils.convert_to_complex(signal, hasN)
    real, imag = data_utils.split_complex(data_utils.normalization(signal, hasN), hasN)
    heatmap(real, axis, 'Real part distribution ' + title_name, show, 'realdisrtibution_' + saved_name)
    heatmap(imag, axis, 'Imag part distribution ' + title_name, show, 'imagdistritbution_' + saved_name)

def complex_diff_fig(img1, img2, axis=None, title_name=None, show=True, saved_name=None):
    '''
    Show real- and imaginary-part difference.
    '''
    signal1, hasN = _check(img1)
    signal2, _ = _check(img2)
    err_real, err_imag = data_utils.complex_diff(img1, img2, True)
    heatmap(err_real, axis, 'Real part difference ratio ' + title_name, show, 'realdiffratio_' + saved_name)
    heatmap(err_imag, axis, 'Imag. part difference ratio ' + title_name, show, 'imagdiffratio_' + saved_name)
    
def phase_diff_fig(img1, img2, axis=None, title_name=None, show=True, saved_name=None):
    '''
    Show pahse difference.
    '''
    img1, _ = _check(img1)
    img2, _ = _check(img2)
    angle_err = analysis.phase_diff(img1, img2)
    heatmap(angle_err, axis, title_name, show, saved_name)
    
        
def project_fig(signal, ref=None, dBgain=0, direction='lateral', axis=None, title_name=None, show=True, saved_name=None):
    '''
    Lateral or axial projection. If `ref` is not None, it also draws the reference signal in the same figure and show
    the beampattern projection difference.
    '''
    signal, hasN = _check(signal)
    fig, ax = plt.subplots(1,1)
    if axis is not None:
        if isinstance(axis, tuple):
            xmin, xmax, zmax, zmin = axis
            axis = np.linspace(xmin,xmax,signal.shape[1]) if direction == 'lateral' else np.linspace(zmin,zmax,signal.shape[0])
        xlabel = 'Lateral position (mm)' if direction == 'lateral' else 'Depth (mm)'
        ax.set_xlabel(xlabel)
        ax.plot(axis, data_utils.projection(signal, direction, dBgain=dBgain, hasN=hasN), color='black')       
    else:
        ax.set_xlabel('Sample')
        ax.plot(data_utils.projection(signal, direction, dBgain=dBgain, hasN=hasN), color='black')
    if ref is not None:
        ref, _ = _check(ref)
        assert ref.shape == signal.shape
        diff = analysis.BPD(signal, ref, direction, dBgain=dBgain)
        if axis is not None:
            ax.plot(axis, data_utils.projection(ref, direction, dBgain=dBgain, hasN=hasN))
        else:
            ax.plot(data_utils.projection(ref, direction, dBgain=dBgain, hasN=hasN))
    else:
        diff = ' '
    title_name = title_name + ' ' + str(diff) if title_name is not None else 'LBPD ' + str(diff) if direction == 'lateral' else 'ABPD ' + str(diff)
    ax.legend(['Prediction','Ground truth'])
    ax.set_ylabel('Normalized intensity (dB)')
    ax.set_title(title_name)
    if saved_name is not None:
        save_fig(fig, saved_name)
    plt.show() if show else plt.close('all')

    
def delay_fig(delay_curve, title_name=None, show=True, saved_name=None):
    '''
    Show delay curve
    '''
    fig = plt.figure()
    plt.plot(delay_curve)
    plt.xlabel('Element')
    plt.ylabel('Delay (pi)')
    plt.title(title_name)
    plt.ylim((-0.5,0.5))
    if saved_name is not None:
        save_fig(fig, saved_name)
    plt.show() if show else plt.close('all')

# ------------------------- displayed according to phase aberration level -------------------------
def err_fig(pred, ref, levels, inds, focus=True, saved_dir=None):
    '''
    This function evaluates the performance of model, indclued mean squared error and error distribution.
    The main idea is the real- and imag-part difference. Analize and display the errors. Finally, the best and
    worst 5 images will be shown.
        Args:
            pred: Numpy array, predicted images.
            ref: Numpy array, ground truth images.
            levels: scalar or vector, phase aberration level.
            inds: scalar or vector, indices of signal1.
            focus: boolean, whether edge is removed.
            model_name: String, saved directory.
            kwargs:
                normalize: boolean, do or not do normalization.
                gain: scalar, gain for log compression.
                vmin: the minimum value of projection,
                    i.e. vmin=0, gain=60 then value<0 would be forced to 0
                    and the max value is 60
    '''
    err, err_2channel = analysis.err_statistic(pred, ref, levels, inds, focus)
    # error for different level
    start = 0
    fig1, ax1 = plt.subplots(1,1)
    fig2, ax2 = plt.subplots(1,1)
    fig3, ax3 = plt.subplots(1,1)
    for level in range(1,constant.k+1):
        level_n_mse = err['MSE'][err['Level'] == level]
        level_n_LBPD = err['LBPD'][err['Level'] == level]
        level_n_ABPD = err['ABPD'][err['Level'] == level]
        end = start + level_n_mse.shape[0] # start level-n index
        # fig1
        ax1.plot(np.arange(start,end), level_n_mse, constant.COLORSHIT[level-1])
        ax1.plot(np.arange(start,end), np.mean(level_n_mse)*np.ones(end-start),'black')
        # fig2
        ax2.plot(np.arange(start, end), level_n_LBPD, constant.COLORSHIT[level-1])
        ax2.plot(np.arange(start, end), np.mean(level_n_LBPD)*np.ones(end-start),'black')
        # fig3
        ax3.plot(np.arange(start, end), level_n_ABPD, constant.COLORSHIT[level-1])
        ax3.plot(np.arange(start, end), np.mean(level_n_ABPD)*np.ones(end-start),'black')
        start = end # end level-n index
        
    ax1.set_title('Mean Squared error')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Intensity')
    
    ax2.set_title('Projected lateral beam profile difference')
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Normalized intensity (dB)')
    
    ax3.set_title('Projected axial beam profile difference')          
    ax3.set_xlabel('Sample')
    ax3.set_ylabel('Normalized intensity (dB)')
    
   
    if saved_dir is not None:
        fig1.savefig(os.path.join(saved_dir, 'msepersample.png'), dpi=300)
        fig2.savefig(os.path.join(saved_dir, 'LBPDpersample.png'), dpi=300)
        fig3.savefig(os.path.join(saved_dir, 'ABPDpersample.png'), dpi=300)
    
    plt.show()
    err = pd.DataFrame(err)
    err_2channel = pd.DataFrame(err_2channel)
    # complex-valued error summation
    boxplot(err, 
            'Level', 
            'maxSE', 
            title_name='Maximum squared error',
            saved_name=os.path.join(saved_dir, 'maxSEboxplot.png') if saved_dir is not None else None)
    # complex-valued max error
    boxplot(err, 
            'Level', 
            'MSE', 
            title_name='Mean squared error',
            saved_name=os.path.join(saved_dir, 'MSEboxplot.png')  if saved_dir is not None else None)
    # 2-branch error summation
    boxplot(err_2channel, 
            'Level', 
            'maxAE', 
            'Channel', 
            title_name='Maximum absolute error of real and imaginary parts',
            saved_name=os.path.join(saved_dir, 'maxAEboxplot.png')  if saved_dir is not None else None)
    # 2-branch max error
    boxplot(err_2channel, 
            'Level', 
            'MAE', 
            'Channel', 
            title_name='Mean absolute error of real and imaginary parts',
            saved_name=os.path.join(saved_dir, 'MAEboxplot.png')  if saved_dir is not None else None)
    # error ratio
    boxplot(err_2channel,
            'Level',
            'AER',
            title_name='Maximum error ratio',
            saved_name=os.path.join(saved_dir, 'ERboxplot.png') if saved_dir is not None else None)
    # lateral projection error
    boxplot(err,
            'Level', 
            'LBPD',
            title_name='Projected lateral beam profile difference',
            saved_name=os.path.join(saved_dir, 'LBPDboxplot.png')  if saved_dir is not None else None)
    # Axial projection error
    boxplot(err, 
            'Level', 
            'ABPD', 
            title_name='Projected axiall beam profile difference',
            saved_name=os.path.join(saved_dir, 'ABPDboxplot.png')  if saved_dir is not None else None)
    # mainlobe pulse difference
    boxplot(err, 
            'Level',
            'MPD', 
            title_name='Mainlobe pulse difference',
            saved_name=os.path.join(saved_dir, 'MPDboxplot.png')  if saved_dir is not None else None)


def leveln_fig(prediction, reference, levels, inds, focus=True, parent_dir=None):
    '''
    Plot from the best to worst lateral or axial beam pattern projection with respect to the different phase
    aberration level.
    
    Args:
        pred: ndarray, prediction.
        ref: ndarray, ground truth.
        levels: scalar or vector, phase aberration level.
        inds: scalar or vector, indices of signal1.
        direction: string, only allow 'lateral' and 'axial'.
        focus: boolean, whether edge is removed.
        **kwargs:
            gain: scalar, gain for log compression.
            vmin: the minimum value of projection,
                i.e. vmin=0, gain=60 then value<0 would be forced to 0
        
    '''
    _, H, W, _ = data_utils.focusing(prediction).shape if focus else prediction.shape
    # lateral projection
    leveln_LP_pred, leveln_LP_ref = analysis.leveln_LP(prediction, reference, levels, focus)
    # axial projection
    leveln_AP_pred, leveln_AP_ref = analysis.leveln_AP(prediction, reference, levels, focus)
    # delay curve
    leveln_delay = analysis.leveln_delay(levels, inds)
    # lateral beam pattern projection difference
    leveln_LBPD = analysis.leveln_LBPD(prediction, reference, levels, focus)
    # axial beam pattern projection difference
    leveln_ABPD = analysis.leveln_ABPD(prediction, reference, levels, focus)
    # mainlobe pulse
    leveln_MP_pred, leveln_MP_ref = analysis.leveln_MP(prediction, reference, levels, inds, focus)
    # mainlobe pulse difference
    leveln_MPD = analysis.leveln_MPD(prediction, reference, levels, inds, focus)
    # absolute error ratio
    leveln_AER_real, leveln_AER_imag = analysis.leveln_cAER(prediction, reference, levels, focus)
    
    level_len = len(np.unique(levels))
    labels = ['level-'+str(ii+1) for ii in range(level_len)]
    fig1, ax1 = plt.subplots(1,1) # level-n lateral projection of prediction
    for level in range(1, level_len+1):
        item_num = len(leveln_LBPD[level-1])
        # indices
        leveln_ind = inds[levels==level]
        leveln_pred = prediction[levels==level]
        leveln_ref = reference[levels==level]
        saved_dir = os.path.join(parent_dir, f'L{level}figures')
        levelind = level - 1
        ax1.plot(np.mean(leveln_LP_pred[levelind],axis=0), color=constant.COLORSHIT[levelind],label=labels[levelind]+' pred.')
        ax1.plot(np.mean(leveln_LP_ref[levelind],axis=0), color=constant.COLORSHIT[levelind],linestyle='dashed', label=labels[levelind]+' ref.')
        for ii in range(item_num):
            axis = info.get_axis(leveln_ind[ii], focus)
            x_axis = np.linspace(axis[0], axis[1], W)
            z_axis = np.linspace(axis[2], axis[3], H)
            z_axis_MP = np.linspace(axis[2], axis[3], 2*H) if info.isbb(prediction) else z_axis
            # LP
            fig = plt.figure()
            plt.plot(x_axis, leveln_LP_pred[levelind][ii], label='Prediction', color='black')
            plt.plot(x_axis, leveln_LP_ref[levelind][ii], label='Ground truth', color='black', linestyle='dashed')
            plt.ylim([-80,0])
            plt.title(f"Level-{level}_No.{leveln_ind[ii]}_LBPD_{leveln_LBPD[levelind][ii]:.2f}") # e.g. level-4_No.129_LP_1.22
            plt.xlabel('Lateral position (mm)')
            plt.ylabel('Normalized intensity (dB)')
            plt.legend()
            save_fig(fig, os.path.join(saved_dir, f'LP_i{leveln_ind[ii]}_L{level}.png'))
            plt.close('all')
            # AP
            fig = plt.figure()
            plt.plot(z_axis, leveln_AP_pred[levelind][ii], label='Prediction', color='black')
            plt.plot(z_axis, leveln_AP_ref[levelind][ii], label='Ground truth', color='black', linestyle='dashed')
            plt.ylim([-80,0])
            plt.title(f"Level-{level}_No.{leveln_ind[ii]}_ABPD_{leveln_ABPD[levelind][ii]:.2f}") # e.g. level-4_No.129_ABPD_1.22
            plt.xlabel('Depth (mm)')
            plt.ylabel('Normalized intensity (dB)')
            plt.legend()
            save_fig(fig, os.path.join(saved_dir, f'AP_i{leveln_ind[ii]}_L{level}.png'))
            plt.close('all')
            # MP
            fig = plt.figure()
            plt.plot(z_axis_MP, leveln_MP_pred[levelind][ii], label='Prediction', color='black')
            plt.plot(z_axis_MP, leveln_MP_ref[levelind][ii], label='Ground truth', color='black', linestyle='dashed')
            plt.title(f"Level-{level}_No.{leveln_ind[ii]}_MPD_{leveln_MPD[levelind][ii]:.4f}") # e.g. level-4_No.129_MPD_1.22
            plt.xlabel('Depth (mm)')
            plt.ylabel('Normalized intensity')
            plt.legend()
            save_fig(fig, os.path.join(saved_dir, f'MP_i{leveln_ind[ii]}_L{level}.png'))
            plt.close('all')
            # AER
            heatmap(leveln_AER_real[levelind][ii], None, f"Level-{level}_No.{leveln_ind[ii]}_RAER", False, os.path.join(saved_dir, f'RAER_i{leveln_ind[ii]}_L{level}.png'), vmax=1, vmin=0)
            heatmap(leveln_AER_imag[levelind][ii], None, f"Level-{level}_No.{leveln_ind[ii]}_IAER", False, os.path.join(saved_dir, f'IAER_i{leveln_ind[ii]}_L{level}.png'), vmax=1, vmin=0)
            # envelope fig
            Bmode_fig(leveln_pred[ii], axis=axis,
                         title_name=f"Level-{level}_No.{leveln_ind[ii]}_prediction",
                         show=False,
                         saved_name=os.path.join(saved_dir, f'predBmode_i{leveln_ind[ii]}_L{level}.png'))
            Bmode_fig(leveln_ref[ii], axis=axis,
                         title_name=f"Level-{level}_No.{leveln_ind[ii]}_reference",
                         show=False,
                         saved_name=os.path.join(saved_dir, f'refBmode_i{leveln_ind[ii]}_L{level}.png'))
            # delay curve
            delay_fig(leveln_delay[levelind][ii],
                      title_name=f"Level-{level}_No.{leveln_ind[ii]}_delay curve",
                      show=False,
                      saved_name=os.path.join(saved_dir, f'delaycurve_i{leveln_ind[ii]}_L{level}.png'))
            info.progressbar(ii+1, item_num, f'Saving level-{level} figures ...')
    # draw and save figure 1 and figure 2
    ax1.set_title('Average projected leateral beam profile')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Normalized intensity (dB)')
    ax1.legend()
    save_fig(fig1, os.path.join(parent_dir, 'avgLP.png'))
    plt.close(1)


def levelnIOU_fig(signal1, signal2, levels, focus=True, saved_dir=None):
    '''
    Draw IOU scores within different regions such as background or main-lobe from weak to strong 
    phase aberration and each level is represented by different colors. The dynamic range is 60dB 
    and divided into four regions included background, far sidelobe, near sidelobe, and main lobe 
    regions. Moreover, each region is account for 20dB.
    
        Args:
            signal1: ndarray, prediction.
            signal2: ndarray, ground truth.
            levels: scalar or vector, phase aberration level.
            inds: scalar or vector, indices of signal1.
            focus: boolean, whether edge is removed.
            
    
    '''
    leveln_iou = analysis.leveln_IOU(signal1, signal2, levels, focus)
    title_names = ['$I \leq -60dB$', '$-60dB < I \leq -40dB$', '$-40dB < I \leq -20dB$', '$-20dB < I \leq 0dB$']
    for iDR in range(leveln_iou.shape[0]):
        start = 0
        fig, ax = plt.subplots(1,1)
        for level in range(leveln_iou.shape[1]):
            end = start + len(leveln_iou[iDR, level]) # number of level-n iou
            ax.scatter(np.arange(start, end), leveln_iou[iDR, level], c=constant.COLORSHIT[level-1])
            start = end
        ax.set_title(title_names[iDR])
        ax.set_xlabel('Sample')
        ax.set_ylabel('IOU score')
        ax.set_ylim((0.0,1.0))
        save_fig(fig, os.path.join(saved_dir, f'IOU_{iDR}'))
        plt.show()

def IOU_fig(signal1, signal2, levels, inds, focus=True, saved_dir=None, DR=60, dBgain=0):
    if focus:
        signal1, signal2 = data_utils.focusing(signal1), data_utils.focusing(signal2)
    iou, DRs, mask1, mask2 = analysis.IOU(signal1, signal2, DR, dBgain)
    title_names = ['$I \leq -60dB$', '$-60dB < I \leq -40dB$', '$-40dB < I \leq -20dB$', '$-20dB < I \leq 0dB$']
    envelope_signal1 = data_utils.log_envelope(signal1, dBgain)
    envelope_signal2 = data_utils.log_envelope(signal2, dBgain)
    # draw iou in different regions for each image
    for ii in range(iou.shape[1]):
        envelope_pred = data_utils.precheck_dim(envelope_signal1[ii], 2, False, False)
        envelope_ref = data_utils.precheck_dim(envelope_signal2[ii], 2, False, False)
        fig = plt.figure(figsize=(20,20))
        plt.subplot(5,2,1)
        plt.imshow(envelope_ref, cmap='gray', vmin=dBgain-DR, vmax=dBgain, aspect='auto')
        plt.title('Ground truth')
        plt.subplot(5,2,2)
        plt.imshow(envelope_pred, cmap='gray', vmin=dBgain-DR, vmax=dBgain, aspect='auto')
        plt.title('Prediction')        
        for iDR in range(len(DRs)):
            plt.subplot(5,2,3+2*iDR)
            plt.imshow(mask2[iDR,ii], cmap='gray', vmin=0, vmax=1, aspect='auto')
            plt.title(title_names[iDR])
            plt.subplot(5,2,4+2*iDR)
            plt.imshow(mask1[iDR,ii], cmap='gray', vmin=0, vmax=1, aspect='auto')
            plt.title(title_names[iDR]  + ' IOU_' + str(round(iou[iDR,ii],2)))
        save_fig(fig, os.path.join(saved_dir, 'IOU', f'IOU_L{levels[ii]}_No.{inds[ii]}.png'))
        plt.close('all')
        info.progressbar(ii+1, iou.shape[1], 'Drawing IOU ...')



