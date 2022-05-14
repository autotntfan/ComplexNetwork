# -*- coding: utf-8 -*-
"""
Created on Wed May  4 21:06:11 2022

@author: benzener
"""
import numpy as np
import scipy.signal as Signal
import matplotlib.pyplot as plt
import os
from scipy    import io
import seaborn as sns
import pandas as pd

class BasedCompute():
    '''
    Basic computation you would utilisze included envelope detection, normalization, 
    clip image, etc.
    '''
    def __init__(self,
                 DATA_SIZE=(2000,257,257),
                 DIR_SAVED=r'./modelinfo',
                 DIR_SIMULATION=r'./simulation_data'):
        
        self.DATA_SIZE = DATA_SIZE # dataset size
        self.DIR_SAVED = DIR_SAVED # saving directory 
        self.DIR_SIMULATION = DIR_SIMULATION # directory of dataset
        
        self._level = None # aberrated level, 1,2,3, or 4.

    def gray(self, img, vmax=1, axis=None, title_name=None, model_name=None, saved_name=None):
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
        self.save_fig(model_name, saved_name)
        plt.show()
    
    def heatmap(self, img, title_name=None, model_name=None, saved_name=None):
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
        self.save_fig(model_name, saved_name)
        plt.show()
        
    def normalization(self, inputs):
        '''
        Limit input range in the range of [-1,1] for coherent signal, or [0,1]
        for incoherent signal.
        
        '''
        shape = inputs.shape
        rank = inputs.ndim
        if rank == 2:
            # [H,W] in real or complex type
            return inputs/np.max(np.abs(inputs))
        elif rank == 3:
            # only [H,W,C], [N,H,W] is NOT available
            if shape[-1]%2:
                # real type array
                return inputs/np.max(np.abs(inputs))
            else:
                # complex type array
                real = inputs[:,:,:shape[-1]//2]
                imag = inputs[:,:,shape[-1]//2:]
                modulus = np.sqrt(real**2 + imag**2)
                return inputs/np.max(modulus)
        elif rank == 4:
            if shape[-1]%2:
                # real type array, e.g. [N,H,W,1]
                return inputs/np.max(np.abs(inputs), axis=(1,2,3), keepdims=True)
            else:
                # complex type array, e.g. [N,H,W,2]
                real = inputs[:,:,:,:shape[-1]//2]
                imag = inputs[:,:,:,shape[-1]//2:]
                modulus = np.sqrt(real**2 + imag**2)
                return inputs/np.max(modulus, axis=(1,2,3), keepdims=True)
        else:
            raise ValueError(f"Unrecognized data type with shape {shape}")
                
    def precheck_dim(self, inputs):
        # expand dim to 4 -> [N,H,W,C]
        if inputs.ndim == 4:
            if inputs.shape[0] != 1:
                raise ValueError('Only support one image')
        elif inputs.ndim == 3:
            inputs = np.expand_dims(inputs, axis=0)
        elif inputs.ndim == 2:
            inputs = inputs.reshape((1,) + inputs.shape + (1,))
        assert inputs.ndim == 4
        return inputs
    
    def reduce_dim(self, inputs):
        # reduce dim to 2 ->[H,W]
        inputs = self.precheck_dim(inputs)
        output_shape = inputs.shape[1:-1]
        return inputs.reshape(output_shape)
    
    def split_complex(self, x):
        '''
        Split complex-valued type or complex-type array to real and imaginary part
            Args:
                x: Numpy array, complex-valued or complex-type.
            Returns:
                Two numpy array represent real and imaginary part respectively with
                the same dimension of inputs.
        '''
        # for complex-valued type array
        if np.iscomplex(x).any():
            return np.real(x), np.imag(x)
        shape = x.shape
        rank = x.ndim
        if rank == 2:
            x = self.convert_to_complex(x)
            return np.real(x), np.imag(x)
        elif rank == 3:
            # for complex-type array
            # only allow [H,W,C]
            if shape[-1]%2:
                x = self.convert_to_complex(x)
                return np.real(x), np.imag(x)
            else:
                return x[:,:,:shape[-1]//2], x[:,:,shape[-1]//2:]
        elif rank == 4:
            if shape[-1]%2:
                x = self.convert_to_complex(x)
                return np.real(x), np.imag(x)
            else:
                return x[:,:,:,:shape[-1]//2], x[:,:,:,shape[-1]//2:]
        else:
            raise ValueError(f'Unrecognized complex array with shape {shape}')
    
    def convert_to_complex(self, inputs):
        '''
        Convert real-valued input to complex-valued type array.
            Args:
                inputs: Numpy array, real-valued or complex-type.
            Return:
                complex-valued array with the same dimension of inputs.
            
        '''
        if np.iscomplex(inputs).any():
            return inputs
        shape = inputs.shape
        rank = inputs.ndim
        if rank == 2:
            return Signal.hilbert(inputs, axis=0)
        elif rank == 3:
            # only allow [H,W,C]
            if shape[-1]%2:
                # real-valued signal needs hilbert transform
                return Signal.hilbert(inputs, axis=0)
            else:
                real, imag = self.split_complex(inputs)
                return real + 1j*imag
        elif rank == 4:
            if shape[-1]%2:
                # real-valued signal needs hilbert transform
                return Signal.hilbert(inputs, axis=1)
            else:
                real, imag = self.split_complex(inputs)
                return real + 1j*imag
        else:
            raise ValueError(f'Unrecognized complex array with shape {shape}')

    
    def envelope_detection(self, signal, DR=None):
        '''
        Detect envelope
            Args:
                signal: Numpy array, it could be real or complex type.
                DR: An integer, dynamic range.
            Return:
                Numpy array in dB scale if DR exists, otherwise linear scale.
        '''
        signal = self.normalization(signal)
        envelope = np.abs(self.convert_to_complex(signal))
        if DR is None:
            return envelope
        else:
            return 20*np.log10(envelope + 1e-16) + DR

    def get_axis(self, img, ind, fs=False):
        '''
        Getting the image axis or smapling rate.
            Args: 
                img: A numpy array. 
                ind: An integer, the index of image.
                fs: Boolean.
            Returns:
                image axis if fs is false, otherwise the sampling rate.
        '''
        img = self.precheck_dim(img)
        H, W = img.shape[1:-1]
        # calculate the aberrated level
        if (ind+1)%4 == 0:
            self._level = 4
        else:
            self._level = (ind+1)%4
        file_name = 'Data_' + str(ind//4 + 1) + '_delay_' + str(self._level) + '.mat'
        print(file_name)
        file_path = os.path.join(self.DIR_SIMULATION, file_name)
        data = io.loadmat(file_path) # reading file gets information
        dx = data.get('dx') * (self.DATA_SIZE[2]/W)
        dz = data.get('dz') * (self.DATA_SIZE[1]/H)
        depth = data.get('depth')/2
        x_axis = np.linspace(0,dx*W-dx,W) * 1e3 # [mm]
        z_axis = np.linspace(0,dz*H-dz,H) * 1e3 + depth * 1e3 # [mm]
        xmin, xmax = (np.min(x_axis), np.max(x_axis))
        zmin, zmax = (np.min(z_axis), np.max(z_axis))
        if fs:
            return 1/(2*dz/1540).reshape([-1])
        else:
            return (xmin, xmax, zmax, zmin)
    
    def angle(self, signal):
        '''
        Compute the angle (phase) for complex value.
            Args:
                signal: Numpy array, complex-valued or real-valued type.
            Return:
                Unwrapping angle
        '''
        complex_signal = self.convert_to_complex(signal)
        wrapped_angle = np.angle(complex_signal)
        if wrapped_angle.ndim == 4:
            return np.apply_over_axes(np.unwrap, wrapped_angle, [1,2])
        else:
            return np.apply_over_axes(np.unwrap, wrapped_angle, [0,1])
        # return wrapped_angle
    
    def check_data_range(self, x, maxv=1, minv=0):
        if np.max(x) > maxv or np.min(x) < minv:
            raise ValueError('Values are not in boundary')
            
    def focusing(self, img, ratio=10):
        '''
        In order to remove artifacts around edge.
            Args:
                img: Numpy array, displayed images.
                ratio: An integer, clipping ratio of image.
            Return:
                Pruned numpy arrays.
        '''
        shape = img.shape
        if img.ndim == 4:
            H, W = shape[1]//ratio, shape[2]//ratio
            return img[:,H:-H,W:-W,:]
        elif img.ndim == 3:
            H, W = shape[0]//ratio, shape[1]//ratio
            return img[H:-H,W:-W,:]
        elif img.ndim == 2:
            H, W = shape[0]//ratio, shape[1]//ratio
            return img[H:-H,W:-W]
        else:
            raise ValueError(f'Unrecognized complex array with shape {shape}')
            
    def save_fig(self, model_name=None, saved_name=None):
        if model_name and saved_name:
            name = os.path.join(self.DIR_SAVED, model_name, model_name + '_' + saved_name + '.png')
            plt.savefig(name, dpi=300)
    
class Difference(BasedCompute):
    '''
    Compute the difference btw two signal, such as complex distribution or angle.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def complex_diff(self, signal1, signal2):
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
        signal1 = self.normalization(signal1)
        signal2 = self.normalization(signal2)
        real1, imag1 = self.split_complex(signal1)
        real2, imag2 = self.split_complex(signal2)
        return np.abs(real1 - real2), np.abs(imag1 - imag2)

    def phase_diff(self, signal1, signal2):
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
        return np.abs(self.angle(signal1) - self.angle(signal2))
            
    def projection(self, signal, ref=None, DR=60, direction='lateral', model_name=None):
        '''
        Lateral projection or axial projection
            Args:
                signal: Numpy array.
                ref: Numpy array, compared signal.
                DR: An integer, dynamic range, the maximum projected value.
                direction: String, only 'lateral' or 'axial'.
                model_name: String, saved directory.
            Returns:
                Show the lateral or axial projection. If the ref exists,
                it draws two projection. Otherwise, only input signal's projection is plotted.
        '''
        if direction not in {'lateral','axial'}:
            raise ValueError("direction only along 'lateral' or 'axial' ")
        if signal.ndim == 4:
            axis = 1 if direction == 'lateral' else 2
        elif signal.ndim < 2 or signal.ndim > 4:
            raise ValueError(f'Unsupport dimension {signal.ndim}')
        else:
            axis = 0 if direction == 'lateral' else 1   
        plt.figure()
        plt.plot(np.max(self.envelope_detection(signal,DR), axis, initial=0))
        if ref is not None:
            assert ref.shape == signal.shape
            plt.plot(np.max(self.envelope_detection(ref,DR), axis, initial=0))
            plt.legend(['pred','true'])
        self.save_fig(model_name, direction)
        plt.show()

            

class Fig(Difference):
    '''
    Show different type of image.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)      

    def envelope_fig(self, img, DR=60, model_name=None, ind=None, saved_name='Bmode'):
        '''
        Show the B-mode image in grayscale.
            Args:
                img: Numpy array, displayed image.
                DR: An integer, dynamic range.
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
        self.gray(img, DR, axis, saved_name, model_name, saved_name)
        
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
        
    def complex_distribution_fig(self, signal, title_='', model_name=None):
        '''
        Show real- and imaginary-part distribution.
        '''
        real, imag = self.split_complex(self.normalization(signal))
        self.heatmap(real, title_ + 'real part distribution', model_name, title_ + 'realdistritbution')
        self.heatmap(imag, title_ + 'imag part distribution', model_name, title_ + 'imagdistritbution')
    
    def complex_diff_fig(self, img1, img2, title_='', DR=None, model_name=None):
        '''
        Show real- and imaginary-part difference.
        '''
        err_real, err_imag = self.complex_diff(img1, img2)
        if DR is not None:
            err_real = 20*np.log10(err_real/2. + 1e-16) + DR
            err_imag = 20*np.log10(err_imag/2. + 1e-16) + DR
            err_real = np.where(err_real<0, 0, err_real)
            err_imag = np.where(err_imag<0, 0, err_imag)
        self.heatmap(err_real, 'real diff ' + str(title_), model_name, 'realpartdiff')
        self.heatmap(err_imag, 'imag diff ' + str(title_), model_name, 'imagpartdiff')
        
    def phase_diff_fig(self, img1, img2, title_='angle difference', threshold=None, model_name=None):
        '''
        Show pahse difference.
        '''
        angle_err = self.reduce_dim(self.phase_diff(img1, img2))
        if threshold is None:
            self.heatmap(angle_err, title_, model_name, 'phasediff')
        else:
            self.heatmap((angle_err<threshold).astype(np.float32), title_, model_name, 'phasebinarydiff')
       
    def error_boxplot(self, pred, truth, OBJ, model_name=None):
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
        levels = []
        for i in range(pred.shape[0]):
            level, _ =  OBJ.find_level(i, train=False)
            levels.append(level)
        err_real_dist, err_imag_dist = Difference().complex_diff(pred, truth)
        err = {
            "level":levels*2,
            "maxerror":np.hstack([np.max(err_real_dist,axis=(1,2,3)),np.max(err_imag_dist,axis=(1,2,3))]),
            'mse':np.hstack([np.sum(err_real_dist,axis=(1,2,3)),np.sum(err_imag_dist,axis=(1,2,3))]),
            "type":np.asarray(['real']*(i+1) + ['imag']*(i+1))
            }
        meanerr = {
            'meanerror':(np.max(err_real_dist,axis=(1,2,3))+np.max(err_imag_dist,axis=(1,2,3)))/2,
            'level':levels
            }
        worst3error = np.argsort(meanerr['meanerror'])[-3:]
        for ind in worst3error:
            self.complex_diff_fig(pred[ind], truth[ind], title_=str(err['level'][ind]))
            self.envelope_fig(pred[ind])
            self.envelope_fig(truth[ind])
        best3error = np.argsort(meanerr['meanerror'])[:3]
        for ind in best3error:
            self.complex_diff_fig(pred[ind], truth[ind], title_=str(err['level'][ind]))
            self.envelope_fig(pred[ind])
            self.envelope_fig(truth[ind])
        df = pd.DataFrame(err)
        plt.figure()
        sns.boxplot(data=df, x='level', y='maxerror', hue='type')
        self.save_fig(model_name, 'maxerrorboxplot')
        plt.show()
        plt.figure()
        sns.boxplot(data=df, x='level', y='mse', hue='type')
        self.save_fig(model_name, 'mseboxplot')
        plt.show()
        plt.figure()
        plt.plot(err['mse'][err['type']=='real'])
        plt.plot(err['mse'][err['type']=='imag'])
        plt.legend(['real','imag'])
        self.save_fig(model_name, 'errordistplot')
        plt.show()
        
        plt.figure()
        sns.boxplot(data=pd.DataFrame(meanerr), x='level', y='meanerror')
        self.save_fig(model_name, 'complexerrordistplot')
        plt.show()
        return err, meanerr
   
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
        self.envelope_fig(self._pred_img, self.DR, self.model_name, self.ind, 'pred'+str(self.ind))
    
    @pred_img.setter
    def pred_img(self, new_img):
        self._pred_img = new_img
        
    @property
    def truth_img(self):
        self.envelope_fig(self._truth_img, self.DR, self.model_name, self.ind, 'truth'+str(self.ind))
        
    @truth_img.setter
    def truth_img(self, new_img):
        self._truth_img = new_img
        
    @property
    def show_complex_dist(self):
        self.complex_distribution(self._pred_img, 'pred', self.model_name)
        self.complex_distribution(self._truth_img, 'truth', self.model_name)
    
    @property
    def show_lateral_projection(self):
        self.projection(self.focusing(self._pred_img), self.focusing(self._truth_img), self.DR, 'lateral', self.model_name)
    
    @property
    def show_axial_projection(self):
        self.projection(self.focusing(self._pred_img), self.focusing(self._truth_img), self.DR, 'axial', self.model_name)
        
    def show_phase_diff(self, threshold=None):
        self.phase_diff_fig(self.focusing(self._pred_img), self.focusing(self._truth_img), 'angle diff ' + str(self.ind), threshold, self.model_name)
        
    def show_complex_diff(self, DR=None):
        self.complex_diff_fig(self.focusing(self._pred_img), self.focusing(self._truth_img), 'complex diff ' + str(self.ind), DR, self.model_name)      

