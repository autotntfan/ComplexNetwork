# -*- coding: utf-8 -*-
"""
Created on Sat May 21 00:09:12 2022

@author: benzener
"""

from computation import Difference
import tensorflow as tf
import numpy as np
import os

class Metrics(Difference):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def mse(self, signal1, signal2, focusing=False, envelope=False, avg=True):
        kwargs = {
            'focusing':focusing,
            'envelope':envelope,
            'avg':avg
            }
        return self.__error_core(signal1, signal2, np.square, **kwargs)
    
    def mae(self, signal1, signal2, focusing=False, envelope=False, avg=True):
        kwargs = {
            'focusing':focusing,
            'envelope':envelope,
            'avg':avg
            }
        return self.__error_core(signal1, signal2, np.abs, **kwargs)
 
    def ssim(self, signal1, signal2, focusing=False, envelope=False, filter_size=7):
        kwargs = {
            'focusing':focusing,
            'envelope':envelope,
            'filter_size':filter_size
            }
        return self.__ssim_core(signal1, signal2, tf.image.ssim, **kwargs)
    
    def ms_ssim(self, signal1, signal2, focusing=False, envelope=False, filter_size=7):
        kwargs = {
            'focusing':focusing,
            'envelope':envelope,
            'filter_size':filter_size
            }
        return self.__ssim_core(signal1, signal2, tf.image.ssim_multiscale, **kwargs)
    
    def save_info(self, signal1, signal2, model_name):
        save_dir = os.path.join(r'./modelinfo', model_name)
        file_name = os.path.join(save_dir, model_name + '_metrics.txt')
        focusing = [False, False, True, True]
        envelope = [False, True, False, True]
        types = ['raw data', 'envelope', 'focusing raw data', 'focusing envelope']
        with open(file_name,'w') as f:
            for ii in range(4):
                f.write('\n' + types[ii] + ' metrics: \n')
                f.write('mse ' + str(self.mse(signal1, signal2, focusing[ii], envelope[ii])) + '\n')
                f.write('mae ' + str(self.mae(signal1, signal2, focusing[ii], envelope[ii])) + '\n')
                f.write('ssim ' + str(self.ssim(signal1, signal2, focusing[ii], envelope[ii])) + '\n')
                f.write('ms_ssim ' + str(self.ms_ssim(signal1, signal2, focusing[ii], envelope[ii])) + '\n')
       
    
    def __ssim_core(self, signal1, signal2, func, **kwargs):
        assert signal1.shape == signal2.shape
        filter_size = kwargs['filter_size']
        kwargs.pop('filter_size')
        signal1, signal2 = self.normalization(signal1), self.normalization(signal2)
        signal1, signal2 = self.__preprocessing(signal1, signal2, **kwargs)
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
    
    def __error_core(self, signal1, signal2, func, **kwargs):
        assert signal1.shape == signal2.shape
        avg = kwargs['avg']
        kwargs.pop('avg')
        signal1, signal2 = self.__preprocessing(signal1, signal2, **kwargs)
        if signal1.ndim == 4 and not avg:
            return np.mean(np.sum(func(signal1 - signal2), axis=(1,2,3)))
        else:
            return np.mean(func(signal1 - signal2))

    def __preprocessing(self, signal1, signal2, focusing=False, envelope=False):
        if focusing:
            signal1 = self.focusing(signal1)
            signal2 = self.focusing(signal2)           
        if envelope:
            signal1 = self.envelope_detection(signal1)
            signal2 = self.envelope_detection(signal2)
        return signal1, signal2
    

        
        
        
        