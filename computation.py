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
    Basic computation you would utilisze included
    normalization: normalize data into a specific range, 
                   e.g. [-1,1] for baseband data.
    precheck_dim: expand dimension to 4, i.e. [H,W] -> [N,H,W,C].
    reduce_dim: squeeze dimension to 2, i.e. [N,H,W,C] -> [H,W].
    split_complex: split complex value x=a+bi to Re{x}=a and Im{x}=b.
    convert_to_complex: convert real value, e.g. RF data, or complex value,
                        e.g. 2-channel BB data, into complex type value a+bi.
    envelope_detection: turn RF data into envelope data with gain or dynamic range.
    get_axis: obtain the physical location of patch.
    get_filename: search the file name of the "i-th" data,
                 e.g. index=93, file_name is Data_24_delay_2.mat.
    get_delaycurve: obtain the i-th data's corresponding delay profile.
    angle: obtain the i-th data's angle distribution in rad.
    check_data_range: check the input data's range is valid.
    focusing: clip the edge to prevent white artifact due 
              to the Gaussian filter in loss function SSIM.
    save_fig: save figure.
    save_info: save model information included parameters.
    read_info: read the file saved by save_info
    projection: obtain lateral or axial projection
    
    last edit 2022/07/01
    
    '''
    def __init__(self,
                 DATA_SIZE=(2000,257,257),
                 DIR_SAVED=r'./modelinfo',
                 DIR_SIMULATION=r'./simulation_data'):
        
        self.DATA_SIZE = DATA_SIZE # dataset size, i.e. RF or BB data size
        self.DIR_SAVED = DIR_SAVED # saving directory
        self.DIR_SIMULATION = DIR_SIMULATION # directory of dataset
        
        self._level = None # aberrated level, 1,2,3, or 4.
        
    def normalization(self, inputs):
        '''
        Limit input in the range of [-1,1] for coherent signal, or [0,1]
        for incoherent signal.
        
        Args:
            inputs: ndarray, in the shape of [N,H,W,C], [H,W,C], [H,W]
        
        Return:
            ndarray, max value = orignal value/abs(max value) along each data
        
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
        # reduce any dim to 2 ->[H,W]
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
        
        Example:
            input = complex type BB data, output = Re{BB}, Im{BB}
            i.e. a+bi -> a,b
            
            input = real type RF data, output = Re{BB}, Im{BB}
            i.e. z -> a,b
            
            input = 2-channel real type BB data, output = first channel, second channel
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
            
        Example:
            input = RF data, output = complex type BB data (a+bi)
            
            input = 2-channel real type BB data, output = complex type BB data (a+bi)
            
            input = complex type BB data, output = complex type BB data
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
        Detect envelope, where gain is equivalent to dynamic range. However, actually
        gain is not equal to DR always. It should be envelope_detection(self, signal, DR, gain).
        
        *The log(0) is undefined, hence that is added by 1e-16, i.e. at least -320 dB.
        
        *Before log compression, data has been normalized since we implement SSIM loss which
        has to limit prediction and reference in the same value range, e.g. [-1,1], but model
        prediction DO NOT pass through loss that value range is not in [-1,1]. Accordingly, here 
        we have to implement normalization to preserve its range is valid and further comparsion
        to be fair.
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
        Getting the physical axis or smapling rate. obtain the
        information about depth, dx, dz, or even sampling rate.
            Args: 
                img: A numpy array. 
                ind: An integer, the index of image.
                fs: Boolean.
            Returns:
                image axis if fs is false, otherwise the sampling rate.
        '''
        img = self.precheck_dim(img)
        H, W = img.shape[1:-1]
        file_name = self.get_filename(ind)
        # calculate the aberrated level
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
        
    def get_filename(self, ind):
        '''
        Arg:
            ind: int, the i-th simulation data       
        Return:
            file_name: string, name of the i-th simulation data
            
        '''
        if (ind+1)%4 == 0:
            self._level = 4
        else:
            self._level = (ind+1)%4
        file_name = 'Data_' + str(ind//4 + 1) + '_delay_' + str(self._level) + '.mat'
        return file_name
        
    def get_delaycurve(self, ind):
        # obtain delay profile
        file_name = self.get_filename(ind)
        file_path = os.path.join(self.DIR_SIMULATION, file_name)
        data = io.loadmat(file_path) # reading file gets information
        delay = data.get('delay_curve')
        if self._level != 1:
            return delay*self._level/8
        else:
            return delay*0;
    
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
            
    def focusing(self, img, ratio=0.05):
        '''
        In order to remove artifacts around edge.
            Args:
                img: Numpy array, displayed images.
                ratio: A decimal, clipping ratio of image.
            Return:
                Pruned numpy arrays.
        '''
        if ratio <= 0 or ratio >= 1:
            raise ValueError('ratio is out of boundary')
        shape = img.shape
        if img.ndim == 4:
            H, W = round(shape[1]*ratio), round(shape[2]*ratio)
            return img[:,H:-H,W:-W,:]
        elif img.ndim == 3:
            H, W = round(shape[0]*ratio), round(shape[1]*ratio)
            return img[H:-H,W:-W,:]
        elif img.ndim == 2:
            H, W = round(shape[0]*ratio), round(shape[1]*ratio)
            return img[H:-H,W:-W]
        else:
            raise ValueError(f'Unrecognized complex array with shape {shape}')
            
    def save_fig(self, model_name=None, saved_name=None, saved_dir=None):
        '''
        save figure to specific path
        Args:
            String
            model_name: model path
            saved_name: desired name to be saved
            saved_dir: desired directory would be built if it does not exist
            
        '''
        if model_name and saved_name:
            # fig is saved only if model_name and saved_name are given
            if saved_dir is None:
                name = os.path.join(self.DIR_SAVED, model_name, model_name + '_' + saved_name + '.png')
            else:
                path = os.path.join(self.DIR_SAVED, model_name, saved_dir)
                if not os.path.exists(path):
                    # if directory doesn't exist then built
                    try:
                        os.mkdir(path)
                    except FileNotFoundError:
                        os.makedirs(path)
                name = os.path.join(path, saved_name + '.png')
            plt.savefig(name, dpi=300)
            
    def save_info(self, model_name, saved_var):
        '''
        save the information of parameters and model, included 
        epoch. seed. and so on
        Args:
            model_name: saved path
            saved_var: parameters to be preserved
        '''
        saved_dir = os.path.join(r'./modelinfo', model_name)
        file_name = model_name + '_parameters.txt'
        if not os.path.exists(saved_dir):
            try:
                os.mkdir(saved_dir)
            except FileNotFoundError:
                os.makedirs(saved_dir)
        saved_path = os.path.join(saved_dir, file_name)
        with open(saved_path, 'w') as f:
            f.write(str(saved_var))
            
    def read_info(self, model_name):
        '''
         reading the information saved by save_info function
         this function is in order to check whether the given
         augments are compatible with loaded model
        '''
        saved_dir = os.path.join(r'./modelinfo', model_name)
        file_name = model_name + '_parameters.txt'
        if not os.path.exists(saved_dir):
            raise FileNotFoundError('file does not exist')
        saved_path = os.path.join(saved_dir, file_name)
        with open(saved_path, 'r') as f:
            content = f.read() # type of content is string
        return eval(content) # convert string to dict
       
    def projection(self, signal, DR=None, direction='lateral', vmin=None):
        '''
        Axial or lateral projection of signal
        Args:
            signal: ndarray, target
            DR: dynamic range
            direction: string, only allow 'lateral' and 'axial'
            vmin: the minimum value of projection,
                i.e. vmin=0, DR=60 then value<0 would be forced to 0
                and the max value is 60
        Return:
            1-D/2-D projection data along axial or lateral determined
            by input sequence.
        '''
        if direction not in {'lateral','axial'}:
            raise ValueError("direction only along 'lateral' or 'axial' ")
        if signal.ndim == 4:
            # [N,H,W,C]
            axis = 1 if direction == 'lateral' else 2
        elif signal.ndim < 2 or signal.ndim > 4:
            raise ValueError(f'Unsupport dimension {signal.ndim}')
        else:
            # [H,W,C] or # [H,W]
            axis = 0 if direction == 'lateral' else 1
        if DR is None:
            # directly applied projection without log compression
            return np.max(signal, axis, initial=vmin)
        else:
            # do log compression depending on DR
            outputs = np.max(self.envelope_detection(signal,DR), axis, initial=vmin)
            return np.squeeze(outputs, -1)
        

class Difference(BasedCompute):
    '''
    Compute the difference btw two signal, such as complex distribution or angle.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def complex_diff(self, signal1, signal2, normalize=True):
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
        # beampattern projection difference
        assert signal1.shape == signal2.shape
        diff = np.abs(self.projection(signal1, DR, *args, **kwargs) - 
                      self.projection(signal2, DR, *args, **kwargs))
        if signal1.ndim == 4:
            return np.mean(diff,axis=1)
        else:
            return np.mean(diff)
    
    def IOU(self, signal1, signal2, DR=60):
        '''
        Calculate the ratio of intersection versus union sets, i.e. IOU = A and B / A or B
        
        Args:
            signal1: Numpy array with shape [N,H,W,C] or [H,W,C]
            signal2: Numpy array with shape [N,H,W,C] or [H,W,C]
            DR: Int, dynmaic range
            
        Return:
            N-by-M ndarray, where DR is divided into M intervals, e.g. DR=70, M=5, included
            ~0, 0~20, 20~40, 40~60, 60~70 dB. Each column contains N samples' IOU value during 
            specific range.
        '''
        if signal1.shape != signal2.shape:
            raise ValueError('Inputs are different size')
        signal1 = self.envelope_detection(signal1, DR)
        signal2 = self.envelope_detection(signal2, DR)
        axis = (1,2,3) if signal1.ndim == 4 else None
        DRs = [20*ii for ii in range(DR//20+1)]
        DRs = DRs + [DR] if DR%20 else DRs
        mask1 = np.zeros((len(DRs),) + signal1.shape)
        mask2 = np.zeros((len(DRs),) + signal1.shape)
        print(mask1.shape)
        for ii, DRmax in enumerate(DRs):
            if DRmax == 0:
                mask1[ii,:] = signal1 < 0
                mask2[ii,:] = signal2 < 0
                iou = np.sum(np.logical_and(signal1 < 0, signal2 < 0), axis=axis) / \
                    np.sum(np.logical_or(signal1 < 0, signal2 < 0), axis=axis)
            else:
                S1 = np.logical_and(signal1 > DRs[ii-1], signal1 <= DRmax)
                S2 = np.logical_and(signal2 > DRs[ii-1], signal2 <= DRmax)
                mask1[ii,:] = S1
                mask2[ii,:] = S2
                iou = np.vstack([iou,np.sum(np.logical_and(S1, S2), axis=axis)/np.sum(np.logical_or(S1, S2), axis=axis)])
        return np.nan_to_num(iou,nan=0.0), DRs, np.squeeze(mask1), np.squeeze(mask2)
    
    def err_statistic(self, signal1, signal2, OBJ, *args, normalize=True, training=False, **kwargs):
        assert signal1.shape == signal2.shape
        levels = np.zeros(signal1.shape[0], int)
        inds = np.zeros(signal1.shape[0], int)
        delay = np.zeros((signal1.shape[0],128))
        for i in range(signal1.shape[0]):
            levels[i], inds[i] = OBJ.find_level(i, train=training)
            delay[i] = self.get_delaycurve(inds[i])
        err_real_dist, err_imag_dist = self.complex_diff(signal1, signal2, normalize)
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
            'ABPD':self.BPD(signal1, signal2, direction='axial', *args, **kwargs),
            'level':levels,
            'ind':inds
            }
        delay = {
            'delay':delay,
            'ind':inds
            }
        
        return err, err_split, delay