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
    import pathlib
    currentpath = os.getcwd()
    path = pathlib.Path(currentpath).parts
    addpath = path[:-2]
    addpath = os.path.join(*addpath)
    if addpath not in sys.path:
        sys.path.append(addpath)
    from baseband.setting import constant
    from baseband.utils.data_utils import reduce_dim, envelope_detection, angle, projection, split_complex, normalization, focusing
    from baseband.utils.info import get_axis, get_level, get_sampling_rate, get_delaycurve, progressbar
    from baseband.utils.analysis import complex_diff, phase_diff, BPD, IOU, err_statistic
    sys.path.remove(addpath)
else:
    from ..setting import constant
    from .data_utils import reduce_dim, envelope_detection, angle, projection, split_complex, normalization, focusing
    from .info import get_axis, get_level, get_sampling_rate, get_delaycurve, progressbar
    from .analysis import complex_diff, phase_diff, BPD, IOU, err_statistic

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
    plt.figure()
    if gain is not None:
        plt.imshow(img, cmap='gray', vmin=gain-DR, vmax=gain, extent=axis, aspect='auto')
    else:
        plt.imshow(img, cmap='gray', vmin=-DR, vmax=0, extent=axis, aspect='auto')
    if title_name is not None:
        plt.title(title_name)
    plt.colorbar()
    save_fig(*args, **kwargs)
    plt.show() if show else plt.close()


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
    plt.show() if show else plt.close()


def boxplot(data, xlabel, ylabel, hue=None, title_name=None, show=True, *args, **kwargs):
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
    save_fig(*args, **kwargs)
    plt.show() if show else plt.close()


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
    plt.show() if show else plt.close()

    
    
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
        signal = focusing(signal)
        ref = focusing(ref)
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
    plt.show() if show else plt.close()

    
def delay_fig(delay, title_name='Delay curve', show=True, *args, **kwargs):
    plt.figure()
    plt.plot(delay)
    plt.xlabel('Element')
    plt.ylabel('Delay (pi)')
    plt.title(title_name)
    plt.ylim((-0.5,0.5))
    save_fig(*args, **kwargs)
    plt.show() if show else plt.close()

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
        colors  = ['pink','hotpink','magenta','m']
        for level in range(1,5):
            level_n_mse = err['sumerr'][err['level'] == level]
            level_n_LBPD = err['LBPD'][err['level'] == level]
            level_n_ABPD = err['ABPD'][err['level'] == level]
            end = start + level_n_mse.shape[0]
            fig1 = plt.figure(1)
            plt.plot(np.arange(start,end), level_n_mse, colors[level-1])
            plt.plot(np.arange(start,end), np.mean(level_n_mse)*np.ones(end-start),'black')
            plt.title('Error summation')
            plt.xlabel('Sample')
            fig2 = plt.figure(2)
            plt.plot(np.arange(start, end), level_n_LBPD, colors[level-1])
            plt.plot(np.arange(start, end), np.mean(level_n_LBPD)*np.ones(end-start),'black')
            plt.title('LBPD difference')
            plt.xlabel('Sample')
            plt.ylabel('LBPD')
            fig3 = plt.figure(3)
            plt.plot(np.arange(start, end), level_n_ABPD, colors[level-1])
            plt.plot(np.arange(start, end), np.mean(level_n_ABPD)*np.ones(end-start),'black')
            plt.title('ABPD difference')          
            plt.xlabel('Sample')
            plt.ylabel('LBPD')
            start = end
        name = os.path.join(constant.MODELPATH, model_name, model_name)
        fig1.savefig(name + '_errordistribution.png', dpi=300)
        fig2.savefig(name + '_LBPDdistribution.png', dpi=300)
        fig3.savefig(name + '_ABPDdistribution.png', dpi=300)
        plt.show()
        err = pd.DataFrame(err)
        err_2channel = pd.DataFrame(err_2channel)
        # complex-valued error summation
        boxplot(err, 
                'level', 
                'sumerr', 
                title_name='Error summation', 
                model_name=model_name, 
                saved_name='complexerrorboxplot')
        # complex-valued max error
        boxplot(err, 
                'level', 
                'maxerr', 
                title_name='Max error', 
                model_name=model_name, 
                saved_name='complexmaxerrorboxplot')
        # 2-branch error summation
        boxplot(err_2channel, 
                'level', 
                'sumerr', 
                'channel', 
                title_name='2-branch error summation', 
                model_name=model_name, 
                saved_name='2Berrorboxplot')
        # 2-branch max error
        boxplot(err_2channel, 
                'level', 
                'maxerr', 
                'channel', 
                title_name='2-branch max error', 
                model_name=model_name, 
                saved_name='2Bmaxerrorboxplot')
        # lateral projection error
        boxplot(err, 
                'level', 
                'LBPD',
                title_name='LBPD', 
                model_name=model_name, 
                saved_name='LBPDboxplot')
        # Axial projection error
        boxplot(err, 
                'level', 
                'ABPD', 
                title_name='ABPD', 
                model_name=model_name, 
                saved_name='ABPDboxplot')

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
                name = 'worst' + key + '_i' + str(err['ind'][ind]) + '_L' + str(err['level'][ind])
            else:
                name = 'best' + key + '_i' + str(err['ind'][ind]) + '_L' + str(err['level'][ind])
            drawfig(pred[ind], ref[ind], delay['delay'][ind], name, **kwargs)
        
def levelnBPD_fig(pred, ref, levels, inds, direction='lateral', focus=True, model_name=None, **kwargs):
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
    assert pred.shape == ref.shape
    if focus:
        pred, ref = focusing(pred), focusing(ref)
    proj_pred, proj_ref = projection(pred, 0), projection(ref, 0)
    delay = np.zeros((pred.shape[0],constant.NELEMENT))
    for ii in range(pred.shape[0]):
        delay[ii] = get_delaycurve(inds[ii])
    BPDs = BPD(pred, ref, direction=direction, **kwargs) # lateral or axial projection
    colors = ['red','green','blue','black']
    labels = ['level1','level2','level3','level4']
    f1 = plt.figure(1)
    ax1 = f1.add_subplot(111)
    f2 = plt.figure(2)
    ax2 = f2.add_subplot(111)
    for level in range(1,5):
        sortinds = np.argsort(BPDs[levels==level]) # sort BPDs of level-n
        Linds = inds[levels==level][sortinds] # level-n index
        Ldelay = delay[levels==level][sortinds] # level-n delay
        Lproj_pred = proj_pred[levels==level][sortinds] # level-n predicted projection
        Lproj_ref = proj_ref[levels==level][sortinds] # level-n reference projection
        LBPDs = BPDs[levels==level][sortinds] # level-n BPD
        Lpred = pred[levels==level][sortinds] # level-n prediction
        Lref = ref[levels==level][sortinds] # level-n reference
        Ldelay = delay[levels==level][sortinds] # level-n delay curve
        ax1.plot(np.mean(Lproj_pred,axis=0), color=colors[level-1],label=labels[level-1])
        ax2.plot(np.mean(Lproj_ref,axis=0), color=colors[level-1],label=labels[level-1])
        
        for ii in range(np.size(LBPDs)):
            dir_ = 'L' + str(level) + 'projection' # saved directory e.g. L4projection
            saved_name = 'L' + str(level) + '_i' + str(Linds[ii]) + '_rank' + str(ii) # saved name e.g. L4_i129
            plt.figure()
            plt.plot(Lproj_pred[ii], label='Prediction')
            plt.plot(Lproj_ref[ii], linestyle='dashed',label='Ground truth')
            plt.title(f"level-{level} {direction} projection_i{inds[ii]}_{LBPDs[ii]:.2f}") # e.g. level-4 lateral projection_i129_1.22
            plt.legend()
            save_fig(model_name, saved_name + 'proj', dir_)
            plt.close()
            envelope_fig(Lpred[ii], 
                         title_name='Prediction Bmode_i' + str(Linds[ii]), 
                         ind=Linds[ii], 
                         model_name=model_name, 
                         saved_name=saved_name+'_Prediction_Bmode',
                         saved_dir=dir_,
                         show=False)
            envelope_fig(Lref[ii], 
                         title_name='Ground truth Bmode_i' + str(Linds[ii]), 
                         ind=Linds[ii], 
                         model_name=model_name, 
                         saved_name=saved_name+'_Groundtruth_Bmode', 
                         saved_dir=dir_,
                         show=False)
            delay_fig(Ldelay[ii], 
                      title_name='Delay curve_i' + str(Linds[ii]), 
                      model_name=model_name, 
                      saved_name=saved_name+'delay', 
                      saved_dir=dir_,
                      show=False)
            progressbar(ii+1, np.size(LBPDs), f'Saving level-{level}')
        # draw PR25, PR50, PR75, PR99 projection of prediction and reference
        plt.figure()
        plt.plot(np.mean(Lproj_pred[:ii//4],axis=0), label='0.25p', color='green')
        plt.plot(np.mean(Lproj_pred[ii//4:2*ii//4], axis=0), label='0.50p', color='blue')
        plt.plot(np.mean(Lproj_pred[2*ii//4:3*ii//4],axis=0), label='0.75p', color='red')
        plt.plot(np.mean(Lproj_pred[3*ii//4:],axis=0), label='1.00p', color='black')
        plt.plot(np.mean(Lproj_ref[:ii//4], axis=0), linestyle='dashed', label='0.25t', color='green')
        plt.plot(np.mean(Lproj_ref[ii//4:2*ii//4], axis=0), linestyle='dashed', label='0.50t', color='blue')
        plt.plot(np.mean(Lproj_ref[2*ii//4:3*ii//4], axis=0), linestyle='dashed', label='0.75t', color='red')
        plt.plot(np.mean(Lproj_ref[3*ii//4:], axis=0), linestyle='dashed', label='1.00t', color='black')
        plt.legend()
        save_fig(model_name, 'L' + str(level) + 'projection performance', dir_)
        plt.close()
    # draw and save figure 1 and figure 2
    ax1.set_title('Prediction')
    ax1.set_xlabel('Lateral position (mm)')
    ax1.set_ylabel('Depth (mm)')
    ax1.legend()
    save_fig(model_name, 'avgLBPDprediction', fig=f1)
    plt.close(1)
    ax2.set_title('Ground truth')
    ax2.set_xlabel('Lateral position (mm)')
    ax2.set_ylabel('Depth (mm)')
    ax2.legend()
    save_fig(model_name, 'avgLBPDreference', fig=f2)
    plt.close(2)

def levelnIOU_fig(signal1, signal2, levels, inds, focus=True, model_name=None):
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
    assert signal1.shape == signal2.shape
    if focus:
        signal1, signal2 = focusing(signal1), focusing(signal2)
    # --------
    gain = 0
    DR = 60
    gap = 20
    # --------
    iou, DRs, mask1, mask2 = IOU(signal1, signal2, DR, gain, gap)
    colors = ['red','green','blue','black']
    title_names = ['I <= -60dB', '-60dB < I <= -40dB', '-40dB < I <= -20dB', '-20dB < I <= 0dB']
    # draw iou in different regions for each image
    for ii in range(iou.shape[1]):
        envelope_pred = reduce_dim(envelope_detection(signal1[ii], gain))
        envelope_ref = reduce_dim(envelope_detection(signal2[ii], gain))
        plt.figure(figsize=(20,20))
        plt.subplot(5,2,1)
        plt.imshow(envelope_pred, cmap='gray', vmin=gain-DR, vmax=gain, aspect='auto')
        plt.title('Prediction')
        plt.subplot(5,2,2)
        plt.imshow(envelope_ref, cmap='gray', vmin=gain-DR, vmax=gain, aspect='auto')
        plt.title('Ground truth')
        for iDR in range(len(DRs)):
            plt.subplot(5,2,3+2*iDR)
            plt.imshow(mask1[iDR,ii], cmap='gray', vmin=0, vmax=1, aspect='auto')
            plt.title(title_names[iDR])
            plt.subplot(5,2,4+2*iDR)
            plt.imshow(mask2[iDR,ii], cmap='gray', vmin=0, vmax=1, aspect='auto')
            plt.title(title_names[iDR] + str(iou[iDR,ii]))
        save_fig(model_name, 'IOU_L' + str(levels[ii]) + '_i' + str(inds[ii]), 'IOU')
        plt.close()
        progressbar(ii+1, iou.shape[1], 'Drawing IOU')
    # draw iou distribution for the whole dataset in a sequence of phase aberration level.
    start = 0
    for level in range(1,5):
        # phase aberration level
        level_n_iou = iou[:,levels==level]
        end = start + level_n_iou.shape[1] # number of level-n iou
        for iDR in range(iou.shape[0]):
            # DR interval
            plt.figure(iDR+1)
            # draw iou distribution under this aberration level for different regions
            plt.scatter(np.arange(start,end),level_n_iou[iDR,:],c=colors[level-1])
            plt.title(title_names[iDR])
            plt.xlabel('Sample')
            plt.ylabel('IOU scores')
            plt.ylim((0.0,1.0))
            save_fig(model_name, 'IOU_' + str(iDR))
        start = end
    plt.show()



# def multilevel_projection_m(pred, truth, OBJ):
#     pred = focusing(pred)
#     truth = focusing(truth)
#     proj_pred = projection(pred, 0)
#     proj_true = projection(truth, 0)
#     levels = np.zeros(pred.shape[0], int)
#     pred_proj = np.zeros((4,pred.shape[2]))
#     true_proj = np.zeros((4,pred.shape[2]))
#     for i in range(pred.shape[0]):
#         levels[i], _ = OBJ.find_level(i, train=False)
#     for i in range(1,5):
#         pred_proj[i-1,:] = np.mean(proj_pred[levels==i],axis=0)
#         true_proj[i-1,:] = np.mean(proj_true[levels==i],axis=0)
#     return pred_proj,true_proj
