# -*- coding: utf-8 -*-
"""
Created on Wed May  4 21:06:11 2022

@author: benzener
"""
import os
from computation import Difference
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Fig(Difference):
    '''
    Show different type of image.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)      

    def gray(self, img, vmax=1, axis=None, title_name=None, *args, **kwargs):
        '''
        Show grayscale image.
            Args:
                img: A numpy array, displayed image has only ONE color channel for grayscale.
                vmax: An integer, upper limint of displayed range
                axis: A tuple, displayed axis consists of (xmin, xmax, ymin, ymax)
                title_name: string, title of plot
                model_name: string, saving directory of which model
                saved_name: string, suffix of saved image. e.g. model_name/model_name+suffix.png
            return:
                ONLY display gray-scale image
                
        '''
        img = self.reduce_dim(img) # reshape to [H,W]
        plt.figure()
        plt.imshow(img, cmap='gray', vmin=0, vmax=vmax, extent=axis, aspect='auto')
        if title_name is not None:
            plt.title(title_name)
        plt.colorbar()
        self.save_fig(*args, **kwargs)
        plt.show()
    
    def heatmap(self, img, title_name=None, *args, **kwargs):
        '''
        Show heatmap image.
            Args:
                img: A numpy array, displayed image has only ONE color channel for grayscale.
                axis: A tuple, displayed axis consists of (xmin, xmax, ymin, ymax)
                title_name: string, title of plot
                model_name: string, saving directory of which model
                saved_name: string, suffix of saved image. e.g. model_name/model_name+suffix.png
            return:
                ONLY display heatmap image
                
        '''
        img = self.reduce_dim(img) # reshape to [H,W]
        plt.figure()
        sns.heatmap(img, cmap='hot')
        if title_name is not None:
            plt.title(title_name)
        self.save_fig(*args, **kwargs)
        plt.show()
    
    def boxplot(self, data, xlabel, ylabel, hue=None, title_name=None, *args, **kwargs):
        plt.figure()
        sns.boxplot(data=data, x=xlabel, y=ylabel, hue=hue)
        if title_name is not None:
            plt.title(title_name)
        self.save_fig(*args, **kwargs)
        plt.show()

    def envelope_fig(self, img, DR=60, title_name='Bmode', ind=None, model_name=None, saved_name='Bmode', *args, **kwargs):
        '''
        Show the B-mode image in grayscale.
            Args:
                img: Numpy array, displayed image.
                DR: An integer, dynamic range.
                title_name: String, figure title
                model_name: String, saved directory.
                ind: An integer, the index of displayed image. In order to get displayed axis.
                saved_name: String, saved name.
        '''
        if ind is not None:
            axis = self.get_axis(img, ind)
        else:
            axis = None
        img = self.envelope_detection(img, DR)
        img = self.reduce_dim(img)
        if self._level is not None:    
            saved_name = saved_name + '_' + str(self._level)
            title_name = title_name + str(self._level)
        self.gray(img, DR, axis, title_name, model_name, saved_name, *args, **kwargs)
        
    def fft_fig(self, signal, ind, Aline=False):
        '''
        Show FFT.
            Args:
                img: Numpy array, analyzed image.
                ind: An integer, the index of displayed image. In order to get sampling rate.
                Aline: Boolean, only compute the center signal.
        '''
        # signal shape = [H,W,C]
        fs = self.get_axis(signal, ind, fs=True)
        # signal shape = [H,W]
        signal = self.reduce_dim(signal)
        if Aline:
            center = signal.shape[1]//2
            SIGNAL = np.abs(np.fft.fftshift(np.fft.fft(signal[:,center], axis=0)))
        else:
            SIGNAL = np.abs(np.fft.fftshift(np.fft.fft(signal, axis=0)))
        freq_axis = np.linspace(-fs/2,fs/2,SIGNAL.shape[0])/1e6
        plt.figure()
        plt.plot(freq_axis,SIGNAL)
        plt.xlabel('MHz')
        plt.show()
        
    def angle_fig(self, signal, model_name=None):
        '''
        Show phase distribution.
        '''
        angle = self.reduce_dim(self.angle(signal))           
        self.heatmap(angle.astype(np.float32), 'angle distribution', model_name, 'phase')
        
    def complex_distribution_fig(self, signal, title_name='', model_name=None):
        '''
        Show real- and imaginary-part distribution.
        '''
        real, imag = self.split_complex(self.normalization(signal))
        self.heatmap(real, title_name + 'real part distribution', model_name, title_name + 'realdistritbution')
        self.heatmap(imag, title_name + 'imag part distribution', model_name, title_name + 'imagdistritbution')
    
    def complex_diff_fig(self, img1, img2, title_name='', DR=None, model_name=None, saved_name='', *args, **kwargs):
        '''
        Show real- and imaginary-part difference.
        '''
        err_real, err_imag = self.complex_diff(img1, img2)
        if DR is not None:
            err_real = 20*np.log10(err_real/2. + 1e-16) + DR
            err_imag = 20*np.log10(err_imag/2. + 1e-16) + DR
            err_real = np.where(err_real<0, 0, err_real)
            err_imag = np.where(err_imag<0, 0, err_imag)
        self.heatmap(err_real, 'real diff ' + str(title_name), model_name, 'realpartdiff' + saved_name, *args, **kwargs)
        self.heatmap(err_imag, 'imag diff ' + str(title_name), model_name, 'imagpartdiff' + saved_name, *args, **kwargs)
        
    def phase_diff_fig(self, img1, img2, title_name='angle difference', threshold=None, model_name=None):
        '''
        Show pahse difference.
        '''
        angle_err = self.reduce_dim(self.phase_diff(img1, img2))
        if threshold is None:
            self.heatmap(angle_err, title_name, model_name, 'phasediff')
        else:
            self.heatmap((angle_err<threshold).astype(np.float32), title_name, model_name, 'phasebinarydiff')
    
    def project_fig(self, signal, ref=None, DR=0, direction='lateral', *args, **kwargs):
        plt.figure()
        plt.plot(self.projection(signal, DR, direction))
        if ref is not None:
            assert ref.shape == signal.shape
            diff = self.BPD(signal, ref, DR, direction=direction)
            plt.plot(self.projection(ref, DR, direction))
            plt.legend(['pred','true'])
            plt.title(str(diff))
        self.save_fig(*args, **kwargs)
        plt.show()
    
    def err_fig(self, pred, truth, OBJ, model_name=None):
        '''
        This function evaluates the performance of model, indclued mean squared error and error distribution.
        The main idea is the real- and imag-part difference. Analize and display the errors. Finally, the best and
        worst 5 images will be shown.
            Args:
                pred: Numpy array, predicted images.
                truth: Numpy array, ground truth images.
                OBJ: Class, in order to find the aberrated level of each image.
                model_name: String, saved directory.
        '''
        pred = self.focusing(pred)
        truth = self.focusing(truth)
        err, err_split = self.err_statistic(pred, truth, OBJ)
        # error for different level
        start = 0
        color  = ['pink','hotpink','magenta','m']
        for level in range(1,5):
            level_n_mse = err['sumerr'][err['level'] == level]
            level_n_LBPD = err['LBPD'][err['level'] == level]
            end = start + level_n_mse.shape[0]
            fig1 = plt.figure(1)
            plt.plot(np.arange(start,end), level_n_mse, color[level-1])
            plt.plot(np.arange(start,end), np.mean(level_n_mse)*np.ones(end-start),'black')
            plt.title('error summation')
            fig2 = plt.figure(2)
            plt.plot(np.arange(start, end), level_n_LBPD, color[level-1])
            plt.plot(np.arange(start, end), np.mean(level_n_LBPD)*np.ones(end-start),'black')
            plt.title('LBPD difference')
            start = end
        name = os.path.join(self.DIR_SAVED, model_name, model_name)
        fig1.savefig(name + '_errorleveldist.png', dpi=300)
        fig2.savefig(name + '_LBPDleveldist.png', dpi=300)
        plt.show()
        err = pd.DataFrame(err)
        err_split = pd.DataFrame(err_split)
        # complex-valued error summation
        plt.figure()
        sns.boxplot(data=err, x='level', y='sumerr')
        plt.title('error summation for complex-valued')
        self.save_fig(model_name, 'complexerrordistplot')
        plt.show()
        # complex-valued max error
        plt.figure()
        sns.boxplot(data=err, x='level', y='maxerr')
        plt.title('max error for complex-valued')
        self.save_fig(model_name, 'complexmaxerrorboxplot')
        plt.show()
        # 2-branch error summation
        plt.figure()
        sns.boxplot(data=err_split, x='level', y='sumerr', hue='type')
        plt.title('2-branch error summation')
        self.save_fig(model_name, 'twoBerrordistplot')
        plt.show()
        # 2-branch max error
        plt.figure()
        sns.boxplot(data=err_split, x='level', y='maxerr', hue='type')
        plt.title('2-branch max error')
        self.save_fig(model_name, 'twoBmaxerrorboxplot')
        plt.show()
        # lateral projection error
        plt.figure()
        sns.boxplot(data=err, x='level', y='LBPD')
        plt.title('LBPD')
        self.save_fig(model_name, 'LBPDboxplot')
        plt.show()
        
    def topn_err_fig(self, pred, truth, OBJ, n=3, model_name=None):
        pred = self.focusing(pred)
        truth = self.focusing(truth)
        err, _ = self.err_statistic(pred, truth, OBJ)
        worstnerror = np.argsort(err['maxerr'])[-n:]
        kwargs = {
            'model_name':model_name,
            'saved_dir':'projection'
            }
        for ind in worstnerror:
            name = 'worstmaxerr' + str(ind) + '_L' + str(err['level'][ind])
            self.envelope_fig(pred[ind], title_name=name + '_p', saved_name=name + '_p', **kwargs)
            self.envelope_fig(truth[ind], title_name=name + '_t', saved_name=name + '_t', **kwargs)
            self.complex_diff_fig(pred[ind], truth[ind], title_name=name, saved_name=name, **kwargs)
            self.project_fig(pred[ind], truth[ind], saved_name=name + '_lateral', **kwargs)
        bestnerror = np.argsort(err['maxerr'])[:n]
        for ind in bestnerror:
            name = 'bestmaxerr' + str(ind) + '_L' + str(err['level'][ind])
            self.envelope_fig(pred[ind], title_name=name + '_p', saved_name=name + '_p', **kwargs)
            self.envelope_fig(truth[ind], title_name=name + '_t', saved_name=name + '_t', **kwargs)
            self.complex_diff_fig(pred[ind], truth[ind], title_name=name, saved_name=name, **kwargs)
            self.project_fig(pred[ind], truth[ind], saved_name=name + '_lateral', **kwargs) 
        worstnerror = np.argsort(err['LBPD'])[-n:]
        for ind in worstnerror:
            name = 'worstLBPD' + str(ind) + '_L' + str(err['level'][ind])
            self.envelope_fig(pred[ind], title_name=name + '_p', saved_name=name + '_p', **kwargs)
            self.envelope_fig(truth[ind], title_name=name + '_t', saved_name=name + '_t', **kwargs)
            self.complex_diff_fig(pred[ind], truth[ind], title_name=name, saved_name=name, **kwargs)
            self.project_fig(pred[ind], truth[ind], saved_name=name + '_lateral', **kwargs)
        
        bestnerror = np.argsort(err['LBPD'])[:n]
        for ind in bestnerror:
            name = 'bestLBPD' + str(ind) + '_L' + str(err['level'][ind])
            self.envelope_fig(pred[ind], title_name=name + '_p', saved_name=name + '_p', **kwargs)
            self.envelope_fig(truth[ind], title_name=name + '_t', saved_name=name + '_t', **kwargs)
            self.complex_diff_fig(pred[ind], truth[ind], title_name=name, saved_name=name, **kwargs)
            self.project_fig(pred[ind], truth[ind], saved_name=name + '_lateral', **kwargs)
   
class VerifyPred(Fig):
    
    def __init__(self,
                 pred_img,
                 truth_img,
                 ind,
                 DR=60,
                 model_name=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.ind = ind
        self.DR = DR
        self.model_name = model_name
        
        self._pred_img = pred_img
        self._truth_img = truth_img
        
    @property
    def pred_img(self):
        self.envelope_fig(self._pred_img, self.DR, 'prediction' , self.ind , self.model_name, 'pred'+str(self.ind))
    
    @pred_img.setter
    def pred_img(self, new_img):
        self._pred_img = new_img
        
    @property
    def truth_img(self):
        self.envelope_fig(self._truth_img, self.DR, 'ground truth', self.ind, self.model_name, 'truth'+str(self.ind))
        
    @truth_img.setter
    def truth_img(self, new_img):
        self._truth_img = new_img
        
    @property
    def show_complex_dist(self):
        self.complex_distribution_fig(self._pred_img, 'pred', self.model_name)
        self.complex_distribution_fig(self._truth_img, 'truth', self.model_name)
    
    @property
    def show_lateral_projection(self):
        self.project_fig(self.focusing(self._pred_img), self.focusing(self._truth_img), self.DR, 'lateral', self.model_name, 'lateral')
    
    @property
    def show_axial_projection(self):
        self.project_fig(self.focusing(self._pred_img), self.focusing(self._truth_img), self.DR, 'axial', self.model_name, 'axial')
        
    def show_phase_diff(self, threshold=None):
        self.phase_diff_fig(self.focusing(self._pred_img), self.focusing(self._truth_img), 'angle diff ' + str(self.ind), threshold, self.model_name)
        
    def show_complex_diff(self, DR=None):
        self.complex_diff_fig(self.focusing(self._pred_img), self.focusing(self._truth_img), 'complex diff ' + str(self.ind), DR, self.model_name)      
