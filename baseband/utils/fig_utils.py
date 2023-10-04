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
    from baseband.utils.data_utils import reduce_dim, envelope_detection, angle, projection, split_complex, normalization, focusing
    from baseband.utils.info import get_axis, get_level, get_sampling_rate, get_delaycurve, progressbar
    from baseband.utils.analysis import complex_diff, phase_diff, BPD, IOU, err_statistic, leveln_IOU, mainlobe_pulse_diff
    sys.path.remove(addpath)
else:
    from ..setting import constant
    from .data_utils import reduce_dim, envelope_detection, angle, projection, split_complex, normalization, focusing
    from .info import get_axis, get_level, get_sampling_rate, get_delaycurve, progressbar
    from .analysis import complex_diff, phase_diff, BPD, IOU, err_statistic, leveln_IOU, mainlobe_pulse_diff
from scipy import signal
from scipy.io import loadmat
# ------------------------- basic figure -------------------------
def save_fig(model_name=None, saved_name=None, saved_dir=None, fig=None):
    '''
    save figure to specific path
    Args:
        String
        model_name: model path
        saved_name: saved figure name
        saved_dir: desired directory will be built if it does not exist
        fig: saved figure
        
    '''
    if model_name and saved_name: 
        # fig is saved only if model_name and saved_name are given
        if saved_dir is None:
            name = os.path.join(constant.MODELPATH, model_name, model_name + '_' + saved_name + '.png')
        else:
            path = os.path.join(constant.MODELPATH, model_name, saved_dir)
            if not os.path.exists(path):
                # if directory doesn't exist then built
                try:
                    os.mkdir(path)
                except FileNotFoundError:
                    os.makedirs(path) # make parent directory
            name = os.path.join(path, saved_name + '.png')
        if fig:
            fig.savefig(name, dpi=300)
        else:
            plt.savefig(name, dpi=300)
        
def gray(img, DR=60, gain=None, axis=None, title_name=None, show=True, *args, **kwargs):
    '''
    Show grayscale image. The default displayed range of color bar under 0. e.g. color range [-60,0] 
    whose dynamic range is 60 and gain is 0. Otherwise, the displayed range of color bar is [gain-DR, gain].
        Args:
            img: A numpy array, displayed image has only ONE color channel for grayscale.
            DR: scalar, dynamic range
            gain: scalar, image displayed gain
            axis: A tuple, displayed axis consists of (xmin, xmax, ymax, ymin) for image
            title_name: string, title of plot
            kwargs:
                model_name: string, saving directory of which model
                saved_name: string, suffix of saved image. e.g. model_name/model_name+suffix.png
                saved_dir: desired directory will be built if it does not exist
        return:
            ONLY display gray-scale image
            
    '''
    img = reduce_dim(img) # reshape to [H,W]
    if gain is not None:
        plt.imshow(img, cmap='gray', vmin=gain-DR, vmax=gain, extent=axis, aspect='auto')
    else:
        plt.imshow(img, cmap='gray', vmin=-DR, vmax=0, extent=axis, aspect='auto')
    if title_name is not None:
        plt.title(title_name)
    plt.colorbar()
    save_fig(*args, **kwargs)
    plt.show() if show else plt.close('all')


def heatmap(img, axis=None, title_name=None, show=True, *args, **kwargs):
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
    img = reduce_dim(img) # reshape to [H,W]
    plt.figure()
    if axis is not None:
        xlabel = np.linspace(axis[0],axis[1],img.shape[1])
        ylabel = np.linspace(axis[3],axis[2],img.shape[0])
        sns.heatmap(img, cmap='hot', xticklabels=xlabel, yticklabels=ylabel)
    else:
        sns.heatmap(img, cmap='hot')
    if title_name is not None:
        plt.title(title_name)
    save_fig(*args, **kwargs)
    plt.show() if show else plt.close('all')


def boxplot(data, xlabel, ylabel, hue=None, title_name=None, show=True, y_label=None, *args, **kwargs):
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
    plt.figure()
    sns.boxplot(data=data, x=xlabel, y=ylabel, hue=hue)
    if title_name is not None:
        plt.title(title_name)
    plt.xlabel('Level')
    if y_label is not None:
        plt.ylabel(y_label)
    save_fig(*args, **kwargs)
    plt.show() if show else plt.close('all')


def envelope_fig(img, 
                 DR=60, 
                 gain=0, 
                 title_name='Bmode', 
                 ind=None, 
                 saved_name='Bmode',
                 show=True,
                 *args, 
                 **kwargs):
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
    axis = None if ind is None else get_axis(img, ind)
    if ind is not None:    
        level = get_level(ind)
        saved_name = saved_name + '_L' + str(level)
        title_name = title_name + '_L' + str(level)
    img = envelope_detection(img, gain)
    img = reduce_dim(img)
    gray(img, DR, gain, axis, title_name, show, *args, saved_name=saved_name, **kwargs)
    
def envelope_fig2(imgs,
                  DR=60,
                  gain=0,
                  title_names='Bmode',
                  ind=None,
                  saved_name='Bmode',
                  show=True,
                  *args,
                  **kwargs):
    plt.figure()
    
    for ii,img in enumerate(imgs):
        plt.subplot(1,2,ii+1)
        axis = None if ind is None else get_axis(img, ind)
        if ind is not None:    
            level = get_level(ind)
            saved_name = saved_name + '_L' + str(level)
            title_name = title_names[ii] + '_L' + str(level)
        img = envelope_detection(img, gain)
        img = reduce_dim(img)
        if gain is not None:
            plt.imshow(img, cmap='gray', vmin=gain-DR, vmax=gain, extent=axis, aspect='auto')
        else:
            plt.imshow(img, cmap='gray', vmin=-DR, vmax=0, extent=axis, aspect='auto')
        if title_names is not None:
            plt.title(title_name)
        plt.colorbar()
    plt.tight_layout()
    save_fig(*args, saved_name=saved_name, **kwargs)
    plt.show() if show else plt.close('all')
        
    
    
def fft_fig(signal, ind, Aline=False, saved_name='Spectrum', model_name=None, show=True):
    '''
    Show FFT.
        Args:
            img: Numpy array, analyzed image.
            ind: An integer, the index of displayed image. In order to get sampling rate.
            Aline: Boolean, only compute the center signal.
    '''
    # signal shape = [H,W,C]
    fs = get_sampling_rate(signal, ind)
    # signal shape = [H,W]
    signal = reduce_dim(signal)
    if Aline:
        # check the spectrum of the central Aline
        center = signal.shape[1]//2
        SIGNAL = np.abs(np.fft.fftshift(np.fft.fft(signal[:,center], axis=0)))
    else:
        SIGNAL = np.abs(np.fft.fftshift(np.fft.fft(signal, axis=0), axes=0))
    freq_axis = np.linspace(-fs/2,fs/2,SIGNAL.shape[0])/1e6
    plt.figure()
    plt.plot(freq_axis,SIGNAL)
    plt.xlabel('MHz')
    plt.title('Spectrum')
    save_fig(model_name,saved_name)
    plt.show() if show else plt.close('all')

    
    
def angle_fig(signal, ind=None, show=True, *args, **kwargs):
    '''
    Show phase distribution.
    '''
    ang = reduce_dim(angle(signal))
    axis = None if ind is None else get_axis(signal, ind)
    heatmap(ang.astype(np.float32), axis, 'Angle distribution', saved_name='phase', show=show, *args, **kwargs)
   
def complex_distribution_fig(signal, ind=None, title_name='', model_name=None, show=True):
    '''
    Show real- and imaginary-part distribution.
    '''
    axis = None if ind is None else get_axis(signal, ind)
    real, imag = split_complex(normalization(signal))
    heatmap(real, axis, 'Real part distribution of ' + title_name, model_name, 'realdistritbution_' + title_name, show)
    heatmap(imag, axis, 'Imag part distribution of ' + title_name, model_name, 'imagdistritbution_' + title_name, show)

def complex_diff_fig(img1, img2, ind=None, title_name='', model_name=None, saved_name='', show=True):
    '''
    Show real- and imaginary-part difference.
    '''
    assert img1.shape == img2.shape
    axis = None if ind is None else get_axis(img1, ind)
    err_real, err_imag = complex_diff(img1, img2)
    heatmap(err_real, axis, 'Real diff ' + str(title_name), model_name, 'realpartdiff_' + saved_name, show)
    heatmap(err_imag, axis, 'Imag diff ' + str(title_name), model_name, 'imagpartdiff_' + saved_name, show)
    
def phase_diff_fig(img1, img2, ind=None, title_name='Angle difference', threshold=None, model_name=None, show=True):
    '''
    Show pahse difference.
    '''
    assert img1.shape == img2.shape
    axis = None if ind is None else get_axis(img1, ind)
    angle_err = reduce_dim(phase_diff(img1, img2))
    if threshold is None:
        heatmap(angle_err, axis, title_name, model_name, 'phasediff', show)
    else:
        heatmap((angle_err<threshold).astype(np.float32), axis, 'Binary ' + title_name, model_name, 'binaryphasediff', show)
        
def project_fig(signal, ref=None, gain=0, direction='lateral', focus=True, show=True, *args, **kwargs):
    '''
    Lateral or axial projection.
    '''
    if focus:
        signal = focusing(signal, hasN=False)
        ref = focusing(ref, hasN=False)
    plt.figure()
    plt.plot(projection(signal, gain, direction))
    if ref is not None:
        assert ref.shape == signal.shape
        diff = BPD(signal, ref, gain, direction=direction)
        plt.plot(projection(ref, gain, direction))
        plt.legend(['Prediction','Ground truth'])
    if direction == 'lateral':
        plt.title('LBPD is ' + str(diff))
    elif direction == 'axial':
        plt.title('ABPD is ' + str(diff))
    else:
        raise ValueError('Direction must be lateral or axial')
    save_fig(*args, **kwargs)
    plt.show() if show else plt.close('all')

    
def delay_fig(delay, title_name='Delay curve', show=True, *args, **kwargs):
    plt.figure()
    plt.plot(delay)
    plt.xlabel('Element')
    plt.ylabel('Delay (pi)')
    plt.title(title_name)
    plt.ylim((-0.5,0.5))
    save_fig(*args, **kwargs)
    plt.show() if show else plt.close('all')

# ------------------------- displayed according to phase aberration level -------------------------
def err_fig(pred, ref, levels, inds, focus=True, model_name=None, **kwargs):
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
        assert pred.shape == ref.shape
        if focus:
            pred, ref = focusing(pred), focusing(ref)
        err, err_2channel, _ = err_statistic(pred, ref, levels, inds, **kwargs)
        # error for different level
        start = 0
        fig1, ax1 = plt.subplots(1,1)
        fig2, ax2 = plt.subplots(1,1)
        fig3, ax3 = plt.subplots(1,1)
        for level in range(1,constant.k+1):
            level_n_mse = err['sumerr'][err['Level'] == level]
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
            
        ax1.set_title('Error summation')
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('Intensity')
        
        ax2.set_title('LBPD difference')
        ax2.set_xlabel('Sample')
        ax2.set_ylabel('Normalized intensity (dB)')
        
        ax3.set_title('ABPD difference')          
        ax3.set_xlabel('Sample')
        ax3.set_ylabel('Normalized intensity (dB)')
            
        name = os.path.join(constant.MODELPATH, model_name, model_name)
        fig1.savefig(name + '_errordistribution.png', dpi=300)
        fig2.savefig(name + '_LBPDdistribution.png', dpi=300)
        fig3.savefig(name + '_ABPDdistribution.png', dpi=300)
        plt.show()
        err = pd.DataFrame(err)
        err_2channel = pd.DataFrame(err_2channel)
        # complex-valued error summation
        boxplot(err, 
                'Level', 
                'sumerr', 
                title_name='Error summation', 
                model_name=model_name, 
                saved_name='complexerrorboxplot')
        # complex-valued max error
        boxplot(err, 
                'Level', 
                'maxerr', 
                title_name='Max error', 
                y_label='Normalized intensity',
                model_name=model_name, 
                saved_name='complexmaxerrorboxplot')
        # 2-branch error summation
        boxplot(err_2channel, 
                'Level', 
                'sumerr', 
                'Channel', 
                title_name='2-branch error summation', 
                y_label='Intensity',
                model_name=model_name, 
                saved_name='2Berrorboxplot')
        # 2-branch max error
        boxplot(err_2channel, 
                'Level', 
                'maxerr', 
                'Channel', 
                title_name='2-branch max error', 
                y_label='Intensity',
                model_name=model_name, 
                saved_name='2Bmaxerrorboxplot')
        # lateral projection error
        boxplot(err, 
                'Level', 
                'LBPD',
                title_name='LBPD', 
                y_label='Normalized intensity (dB)',
                model_name=model_name, 
                saved_name='LBPDboxplot')
        # Axial projection error
        boxplot(err, 
                'Level', 
                'ABPD', 
                title_name='ABPD', 
                y_label='Normalized intensity (dB)',
                model_name=model_name, 
                saved_name='ABPDboxplot')
        # mainlobe pulse difference
        boxplot(err, 
                'Level', 
                'pulsediff', 
                title_name='Mainlobe pulse difference', 
                y_label='Intensity',
                model_name=model_name, 
                saved_name='pulsediffboxplot')

def bwp_fig(pred, ref, levels, inds, focus=True, n=3, model_name=None, **kwargs):
    '''
    Show best and worst prediction according to L2 norm and beam pattern projection difference.
    kwargs:
        normalize: boolean, do or not do normalization.
        gain: scalar, gain for log compression.
        vmin: the minimum value of projection,
            i.e. vmin=0, gain=60 then value<0 would be forced to 0
            and the max value is 60
    '''
    assert pred.shape == ref.shape
    if focus:
        pred = focusing(pred)
        ref = focusing(ref)
    err, _ , delay = err_statistic(pred, pred, levels, inds, **kwargs)
    kwargs = {
        'model_name':model_name,
        'saved_dir':'BestWorstPerformance'
        }
    
    def drawfig(data1, data2, delaycurve, name, **kwargs):
        envelope_fig(data1, title_name=name + '_pred', saved_name=name + '_pred', **kwargs)
        envelope_fig(data2, title_name=name + '_ref', saved_name=name + '_ref', **kwargs)
        complex_diff_fig(data1, data2, title_name=name, saved_name=name, **kwargs)
        project_fig(data1, data2, saved_name=name + '_lateralprojection', **kwargs)
        delay_fig(delaycurve, name, saved_name=name + 'delaycurve', **kwargs)
        
    keys = ['sumerr', 'LBPD', 'ABPD']
    for key in keys:
        keyinds = np.hstack([np.argsort(err[key])[:n], np.argsort(err[key])[-n:]])
        for ii, ind in enumerate(keyinds):
            if ii < n:
                name = 'worst' + key + '_i' + str(err['ind'][ind]) + '_L' + str(err['Level'][ind])
            else:
                name = 'best' + key + '_i' + str(err['ind'][ind]) + '_L' + str(err['Level'][ind])
            drawfig(pred[ind], ref[ind], delay['delay'][ind], name, **kwargs)

def leveln_fig(prediction, reference, levels, inds, focus=True, model_name=None, **kwargs):
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
    assert prediction.shape == reference.shape
    H, W = prediction.shape[1], prediction.shape[2]
    if focus:
        pred, ref = focusing(prediction), focusing(reference)
    # lateral projection
    lateral_proj_pred, lateral_proj_ref = projection(pred, 0), projection(ref, 0)
    # axial projection
    axial_proj_pred, axial_proj_ref = projection(pred, 0, 'axial'), projection(ref, 0, 'axial')
    # delay curve
    delay_curve = np.zeros((pred.shape[0],constant.NELEMENT))
    for ii in range(pred.shape[0]):
        delay_curve[ii] = get_delaycurve(inds[ii])
    # lateral beam pattern projection difference
    LBPDs = BPD(pred, ref, direction='lateral', **kwargs) 
    # axial beam pattern projection difference
    ABPDs = BPD(pred, ref, direction='axial', **kwargs)
    # main lobe pulse difference
    pulse_diff, rf_pred_mainlobe_aline, rf_ref_mainlobe_aline = mainlobe_pulse_diff(normalization(pred), ref, inds, True)
    level_len = len(np.unique(levels))
    labels = ['level-'+str(ii+1) for ii in range(level_len)]
    fig1, ax1 = plt.subplots(1,1) # level-n lateral projection of prediction
    
    factor = 2
    pulse_file = loadmat('RFpulses.mat')
    estimated_pulse = pulse_file.get('pulses')
    estimated_pulse = estimated_pulse[inds,1::2]
    if focus:
        estimated_pulse = focusing(estimated_pulse)
    estimated_pulse = signal.resample_poly(estimated_pulse, factor, 1,axis=1)
    rf_pred_mainlobe_aline = signal.resample_poly(rf_pred_mainlobe_aline, factor, 1, axis=1)
    rf_ref_mainlobe_aline = signal.resample_poly(rf_ref_mainlobe_aline, factor, 1, axis=1)

    # fig2, ax2 = plt.subplots(1,1) # level-n lateral projection of reference
    for level in range(1,level_len+1):
        sorted_ind = np.argsort(LBPDs[levels==level])
        leveln_inds = levels==level # level-n sorted index
        leveln_data_inds = inds[leveln_inds][sorted_ind]
        leveln_delay_curve = delay_curve[leveln_inds][sorted_ind] # level-n delay
        leveln_lateral_proj_pred = lateral_proj_pred[leveln_inds][sorted_ind] # level-n predicted projection
        leveln_lateral_proj_ref = lateral_proj_ref[leveln_inds][sorted_ind] # level-n reference projection
        leveln_axial_proj_pred = axial_proj_pred[leveln_inds][sorted_ind] # level-n predicted projection
        leveln_axial_proj_ref = axial_proj_ref[leveln_inds][sorted_ind]
        leveln_LBPD = LBPDs[leveln_inds][sorted_ind]  # level-n LBPD
        leveln_ABPD = ABPDs[leveln_inds][sorted_ind]  # level-n ABPD
        leveln_pred = pred[leveln_inds][sorted_ind]  # level-n prediction
        leveln_ref = ref[leveln_inds][sorted_ind] # level-n reference
        leveln_pulse_diff = pulse_diff[leveln_inds][sorted_ind]
        leveln_rf_pred_mainlobe_aline = rf_pred_mainlobe_aline[leveln_inds][sorted_ind]
        leveln_rf_ref_mainlobe_aline = rf_ref_mainlobe_aline[leveln_inds][sorted_ind]
        leveln_estimated_pulse = estimated_pulse[leveln_inds][sorted_ind]
        ax1.plot(np.mean(leveln_lateral_proj_pred,axis=0), color=constant.COLORSHIT[level-1],label=labels[level-1]+' pred.')
        ax1.plot(np.mean(leveln_lateral_proj_ref,axis=0), color=constant.COLORSHIT[level-1],linestyle='dashed', label=labels[level-1]+' ref.')
        
        for ii in range(len(sorted_ind)):
            axis = get_axis(prediction[ii], leveln_data_inds[ii]) # to get the original axis (without focusing)
            x_axis = np.linspace(axis[0], axis[1], W)
            z_axis = np.linspace(axis[2], axis[3], H)
            if focus:
                x_axis = focusing(x_axis)
                z_axis = focusing(z_axis)
            xf_axis = signal.resample_poly(x_axis, factor, 1, padtype='minimum')
            zf_axis = signal.resample_poly(z_axis, factor, 1, padtype='minimum')
            dir_ = 'L' + str(level) + 'projection' # saved directory e.g. L4projection
            saved_name = 'L' + str(level) + '_i' + str(leveln_data_inds[ii]) + '_rank' + str(ii) # saved name e.g. L4_i129
            plt.figure()
            plt.plot(x_axis, leveln_lateral_proj_pred[ii], label='Prediction')
            plt.plot(x_axis, leveln_lateral_proj_ref[ii], linestyle='dashed',label='Ground truth')
            plt.ylim([-80,0])
            plt.title(f"level-{level} lateral projection_i{leveln_data_inds[ii]}_{leveln_LBPD[ii]:.2f}") # e.g. level-4 lateral projection_i129_1.22
            plt.xlabel('Lateral position (mm)')
            plt.ylabel('Normalized intensity (dB)')
            plt.legend()
            save_fig(model_name, saved_name + 'lateralproj', dir_)
            plt.close('all')
            plt.figure()
            plt.plot(leveln_axial_proj_pred[ii], label='Prediction')
            plt.plot(leveln_axial_proj_ref[ii], linestyle='dashed',label='Ground truth')
            plt.xlabel('Lateral position (mm)')
            plt.ylabel('Normalized intensity (dB)')
            plt.title(f"level-{level} axial projection_i{leveln_data_inds[ii]}_{leveln_ABPD[ii]:.2f}") # e.g. level-4 lateral projection_i129_1.22
            plt.legend()
            save_fig(model_name, saved_name + 'axialproj', dir_)
            plt.close('all')
            plt.figure()
            plt.plot(zf_axis, leveln_rf_pred_mainlobe_aline[ii], label='Prediction')
            plt.plot(zf_axis, leveln_estimated_pulse[ii], label='Estimated pulse')
            plt.plot(zf_axis, leveln_rf_ref_mainlobe_aline[ii], linestyle='dashed',label='Ground truth')
            plt.title(f"level-{level} main-lobe pulse difference_i{leveln_data_inds[ii]}_{leveln_pulse_diff[ii]:.2f}") # e.g. level-4 main-lobe pulse difference_i129_1.22
            plt.xlabel('Depth (mm)')
            plt.ylabel('Normalized intensity')
            plt.legend()
            save_fig(model_name, saved_name + 'pulsediff3', dir_)
            plt.close('all')
            plt.figure()
            plt.plot(zf_axis, leveln_rf_pred_mainlobe_aline[ii], label='Prediction')
            plt.plot(zf_axis, leveln_rf_ref_mainlobe_aline[ii], linestyle='dashed',label='Ground truth')
            plt.title(f"level-{level} main-lobe pulse difference_i{leveln_data_inds[ii]}_{leveln_pulse_diff[ii]:.2f}") # e.g. level-4 main-lobe pulse difference_i129_1.22
            plt.xlabel('Depth (mm)')
            plt.ylabel('Normalized intensity')
            plt.legend()
            save_fig(model_name, saved_name + 'pulsediff', dir_)
            plt.close('all')
            # envelope_fig2((leveln_ref[ii],leveln_pred[ii]),
            #               title_names=['Ground truth Bmode_i' + str(leveln_data_inds[ii]),'Prediction Bmode_i' + str(leveln_data_inds[ii])], 
            #               ind=leveln_data_inds[ii], 
            #               model_name=model_name, 
            #               saved_name=saved_name+'_RefvsPred_Bmode',
            #               saved_dir=dir_,
            #               show=False)
            envelope_fig(leveln_pred[ii], 
                          title_name='Prediction Bmode_i' + str(leveln_data_inds[ii]), 
                          ind=leveln_data_inds[ii], 
                          model_name=model_name, 
                          saved_name=saved_name+'_Prediction_Bmode',
                          saved_dir=dir_,
                          show=False)
            envelope_fig(leveln_ref[ii], 
                          title_name='Ground truth Bmode_i' + str(leveln_data_inds[ii]), 
                          ind=leveln_data_inds[ii], 
                          model_name=model_name, 
                          saved_name=saved_name+'_Groundtruth_Bmode', 
                          saved_dir=dir_,
                          show=False)
            delay_fig(leveln_delay_curve[ii], 
                      title_name='Delay curve_i' + str(leveln_data_inds[ii]), 
                      model_name=model_name, 
                      saved_name=saved_name+'delay', 
                      saved_dir=dir_,
                      show=False)
            progressbar(ii+1, len(sorted_ind), f'Saving level-{level}')
        # draw PR25, PR50, PR75, PR99 projection of prediction and reference
        plt.figure()
        plt.plot(np.mean(leveln_lateral_proj_pred[:ii//4],axis=0), label='0.25p', color='green')
        plt.plot(np.mean(leveln_lateral_proj_pred[ii//4:2*ii//4], axis=0), label='0.50p', color='blue')
        plt.plot(np.mean(leveln_lateral_proj_pred[2*ii//4:3*ii//4],axis=0), label='0.75p', color='red')
        plt.plot(np.mean(leveln_lateral_proj_pred[3*ii//4:],axis=0), label='1.00p', color='black')
        plt.plot(np.mean(leveln_lateral_proj_ref[:ii//4], axis=0), linestyle='dashed', label='0.25t', color='green')
        plt.plot(np.mean(leveln_lateral_proj_ref[ii//4:2*ii//4], axis=0), linestyle='dashed', label='0.50t', color='blue')
        plt.plot(np.mean(leveln_lateral_proj_ref[2*ii//4:3*ii//4], axis=0), linestyle='dashed', label='0.75t', color='red')
        plt.plot(np.mean(leveln_lateral_proj_ref[3*ii//4:], axis=0), linestyle='dashed', label='1.00t', color='black')
        plt.legend()
        save_fig(model_name, 'L' + str(level) + 'projection performance', dir_)
        plt.close('all')
    # draw and save figure 1 and figure 2
    ax1.set_title('Average leateral beam pattern projection')
    ax1.set_xlabel('Lateral position (mm)')
    ax1.set_ylabel('Depth (mm)')
    ax1.legend()
    save_fig(model_name, 'avgLBPprediction', fig=fig1)
    plt.close(1)
    # ax2.set_title('Ground truth')
    # ax2.set_xlabel('Lateral position (mm)')
    # ax2.set_ylabel('Depth (mm)')
    # ax2.legend()
    # save_fig(model_name, 'avgLBPDreference', fig=fig2)
    # plt.close(2)

def levelnIOU_fig(signal1, signal2, levels, model_name=None):
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
    leveln_iou = leveln_IOU(signal1, signal2, levels)
    
    title_names = ['I <= -60dB', '-60dB < I <= -40dB', '-40dB < I <= -20dB', '-20dB < I <= 0dB']
    for iDR in range(leveln_iou.shape[0]):
        start = 0
        fig, ax = plt.subplots(1,1)
        for level in range(leveln_iou.shape[1]):
            end = start + len(leveln_iou[iDR, level]) # number of level-n iou
            ax.scatter(np.arange(start, end), leveln_iou[iDR, level], c=constant.COLORSHIT[level-1])
            start = end
        ax.set_title(title_names[iDR])
        ax.set_xlabel('Sample')
        ax.set_ylabel('IOU scores')
        ax.set_ylim((0.0,1.0))
        save_fig(model_name, 'IOU_' + str(iDR), fig=fig)
        plt.show()

def IOU_fig(signal1, signal2, levels, inds, focus=True, model_name=None, DR=60, gain=0):
    assert signal1.shape == signal2.shape
    if focus:
        signal1, signal2 = focusing(signal1), focusing(signal2)
    iou, DRs, mask1, mask2 = IOU(signal1, signal2, DR, gain)
    title_names = ['I <= -60dB', '-60dB < I <= -40dB', '-40dB < I <= -20dB', '-20dB < I <= 0dB']
    # draw iou in different regions for each image
    for ii in range(iou.shape[1]):
        envelope_pred = reduce_dim(envelope_detection(signal1[ii], gain))
        envelope_ref = reduce_dim(envelope_detection(signal2[ii], gain))
        plt.figure(figsize=(20,20))
        plt.subplot(5,2,1)
        plt.imshow(envelope_ref, cmap='gray', vmin=gain-DR, vmax=gain, aspect='auto')
        plt.title('Ground truth')
        plt.subplot(5,2,2)
        plt.imshow(envelope_pred, cmap='gray', vmin=gain-DR, vmax=gain, aspect='auto')
        plt.title('Prediction')        
        for iDR in range(len(DRs)):
            plt.subplot(5,2,3+2*iDR)
            plt.imshow(mask2[iDR,ii], cmap='gray', vmin=0, vmax=1, aspect='auto')
            plt.title(title_names[iDR]  + ' - ' + str(round(iou[iDR,ii],2)))
            plt.subplot(5,2,4+2*iDR)
            plt.imshow(mask1[iDR,ii], cmap='gray', vmin=0, vmax=1, aspect='auto')
            plt.title(title_names[iDR])
        save_fig(model_name, 'IOU_L' + str(levels[ii]) + '_i' + str(inds[ii]), 'IOU')
        plt.close('all')
        progressbar(ii+1, iou.shape[1], 'Drawing IOU')



