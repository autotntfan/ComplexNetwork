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
            
    def save_fig(self, model_name=None, saved_name=None, saved_dir=None):
        if model_name and saved_name:
            if saved_dir is None:
                name = os.path.join(self.DIR_SAVED, model_name, model_name + '_' + saved_name + '.png')
            else:
                path = os.path.join(self.DIR_SAVED, model_name, saved_dir)
                if not os.path.exists(path):
                    try:
                        os.mkdir(path)
                    except FileNotFoundError:
                        os.makedirs(path)
                name = os.path.join(path, saved_name + '.png')
            plt.savefig(name, dpi=300)
            
    def kaverage(self, x, k=3):
        residual = x[:k]
        return np.hstack([residual, np.convolve(x, np.ones((k,))/k)])

    def projection(self, signal, DR=None, direction='lateral', vmin=None):
        if direction not in {'lateral','axial'}:
            raise ValueError("direction only along 'lateral' or 'axial' ")
        if signal.ndim == 4:
            axis = 1 if direction == 'lateral' else 2
        elif signal.ndim < 2 or signal.ndim > 4:
            raise ValueError(f'Unsupport dimension {signal.ndim}')
        else:
            axis = 0 if direction == 'lateral' else 1
        if DR is None:
            return np.max(signal, axis, initial=vmin)
        else:
            return np.max(self.envelope_detection(signal,DR), axis, initial=vmin)

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
    
    def BPD(self, signal1, signal2, DR=0, *args, **kwargs):
        assert signal1.shape == signal2.shape
        diff = np.abs(self.projection(signal1, DR, *args, **kwargs) -
                      self.projection(signal2, DR, *args, **kwargs))
        if signal1.ndim == 4:
            return np.mean(diff, axis=(1,2))
        else:
            return np.mean(diff)
    
    def err_statistic(self, signal1, signal2, OBJ, *args, training=False, **kwargs):
        assert signal1.shape == signal2.shape
        levels = np.zeros(signal1.shape[0], int)
        inds = np.zeros(signal1.shape[0], int)
        for i in range(signal1.shape[0]):
            levels[i], inds[i] = OBJ.find_level(i, train=training)
        err_real_dist, err_imag_dist = self.complex_diff(signal1, signal2)
        err_split = {
            'level':np.tile(levels,2),
            'maxerr':np.hstack([np.max(err_real_dist,axis=(1,2,3)),np.max(err_imag_dist,axis=(1,2,3))]),
            'sumerr':np.hstack([np.sum(err_real_dist,axis=(1,2,3)),np.sum(err_imag_dist,axis=(1,2,3))]),
            'type':np.asarray(['real']*(i+1) + ['imag']*(i+1)),
            'ind':np.tile(inds,2)
            }
        err = {
            'maxerr':(np.max(err_real_dist,axis=(1,2,3))+np.max(err_imag_dist,axis=(1,2,3)))/2,
            'sumerr':(np.sum(err_real_dist,axis=(1,2,3)) + np.sum(err_real_dist,axis=(1,2,3)))/2,
            'LBPD':self.BPD(signal1, signal2, *args, **kwargs),
            'level':levels
            }
        return err, err_split