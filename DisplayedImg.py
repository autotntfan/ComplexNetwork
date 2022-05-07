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
    
    def __init__(self,
                 DATA_SIZE=(2000,257,257),
                 DIR_SAVED=r'./modelinfo',
                 DIR_SIMULATION=r'./simulation_data'):
        
        self.DATA_SIZE = DATA_SIZE
        self.DIR_SAVED = DIR_SAVED
        self.DIR_SIMULATION = DIR_SIMULATION
        
        self._level = None

    def gray(self, img, vmax=1, axis=None, title_name=None, model_name=None, saved_name=None):
        img = self.reduce_dim(img)
        plt.figure()
        plt.imshow(img, cmap='gray', vmin=0, vmax=vmax, extent=axis, aspect='auto')
        if title_name is not None:
            plt.title(title_name)
        self.save_fig(model_name, saved_name)
        plt.colorbar()
        plt.show()
    
    def heatmap(self, img, title_name=None, model_name=None, saved_name=None):
        img = self.reduce_dim(img)
        plt.figure()
        sns.heatmap(img, cmap='hot')
        if title_name is not None:
            plt.title(title_name)
        self.save_fig(model_name, saved_name)
        plt.show()
        
    def normalization(self, inputs):
        shape = inputs.shape
        rank = inputs.ndim
        if rank == 2:
            # [H,W] in real or complex type
            return inputs/np.max(np.abs(inputs))
        elif rank == 3:
            # only [H,W,C], [N,H,W] is Not available
            if shape[-1]%2:
                # real value
                return inputs/np.max(np.abs(inputs))
            else:
                real = inputs[:,:,:shape[-1]//2]
                imag = inputs[:,:,shape[-1]//2:]
                modulus = np.sqrt(real**2 + imag**2)
                return inputs/np.max(modulus)
        elif rank == 4:
            if shape[-1]%2:
                # real [N,H,W,C]
                return inputs/np.max(np.abs(inputs), axis=(1,2,3), keepdims=True)
            else:
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
        # convert to 2-D
        inputs = self.precheck_dim(inputs)
        output_shape = inputs.shape[1:-1]
        return inputs.reshape(output_shape)
    
    def split_complex(self, x):
        shape = x.shape
        rank = x.ndim
        if rank == 3:
            # only allow [H,W,C]
            if shape[-1]%2:
                raise ValueError(f'Last channel must be odd for real array but get {shape[-1]}')
            else:
                return x[:,:,:shape[-1]//2], x[:,:,shape[-1]//2:]
        elif rank == 4:
            if shape[-1]%2:
                raise ValueError(f'Last channel must be odd for real array but get {shape[-1]}')
            else:
                return x[:,:,:,:shape[-1]//2], x[:,:,:,shape[-1]//2:]
        else:
            raise ValueError(f'Unrecognized complex array with shape {shape}')
    
    def convert_to_complex(self, inputs):
        shape = inputs.shape
        rank = inputs.ndim
        if rank == 3:
            # only allow [H,W,C]
            if shape[-1]%2:
                return Signal.hilbert(inputs, axis=0)
            else:
                real, imag = self.split_complex(inputs)
                return real + 1j*imag
        elif rank == 4:
            if shape[-1]%2:
                return Signal.hilbert(inputs, axis=1)
            else:
                real, imag = self.split_complex(inputs)
                return real + 1j*imag
        else:
            raise ValueError(f'Unrecognized complex array with shape {shape}')

    
    def envelope_detection(self, signal, DR=None):
        signal = self.normalization(signal)
        envelope = np.abs(self.convert_to_complex(signal))
        if DR is None:
            return envelope
        else:
            return 20*np.log10(envelope + 1e-16) + DR

    def get_axis(self, img, ind, fs=False):
        img = self.precheck_dim(img)
        H, W = img.shape[1:-1]
        if (ind+1)%4 == 0:
            self._level = 4
        else:
            self._level = (ind+1)%4
        file_name = 'Data_' + str(ind//4 + 1) + '_delay_' + str(self._level) + '.mat'
        print(file_name)
        file_path = os.path.join(self.DIR_SIMULATION, file_name)
        data = io.loadmat(file_path)
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
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def complex_diff(self, signal1, signal2):
        if signal1.shape != signal2.shape:
            raise ValueError('Inputs are different size')
        signal1 = self.normalization(signal1)
        signal2 = self.normalization(signal2)
        real1, imag1 = self.split_complex(signal1)
        real2, imag2 = self.split_complex(signal2)
        return np.abs(real1 - real2), np.abs(imag1 - imag2)

    def phase_diff(self, signal1, signal2):
        if signal1.shape != signal2.shape:
            raise ValueError('Inputs are different size')
        return self.reduce_dim(np.abs(self.angle(signal1) - self.angle(signal2)))
            
    def projection(self, signal, ref=None, DR=60, direction='lateral', model_name=None):
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
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)      

    def envelope_fig(self, img, DR=60, model_name=None, ind=None, saved_name='Bmode'):
        if ind is not None:
            axis = self.get_axis(img, ind)
        else:
            axis = None
        img = self.envelope_detection(img, DR)
        img = self.reduce_dim(img)
        saved_name = saved_name + '_' + str(self._level)
        self.gray(img, DR, axis, saved_name, model_name, saved_name)
        
    def fft_fig(self, signal, ind, Aline=False):
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
        angle = self.reduce_dim(self.angle(signal))           
        self.heatmap(angle.astype(np.float32), 'angle distribution', model_name, 'phase')
        
    def complex_distribution_fig(self, signal, title_='', model_name=None):
        real, imag = self.split_complex(self.normalization(signal))
        self.heatmap(real, title_ + 'real part distribution', model_name, title_ + 'realdistritbution')
        self.heatmap(imag, title_ + 'imag part distribution', model_name, title_ + 'imagdistritbution')
    
    def complex_diff_fig(self, img1, img2, title_='complexdiff', DR=None, model_name=None):
        err_real, err_imag = self.complex_diff(img1, img2)
        if DR is not None:
            err_real = 20*np.log10(err_real/2. + 1e-16) + DR
            err_imag = 20*np.log10(err_imag/2. + 1e-16) + DR
            err_real = np.where(err_real<0, 0, err_real)
            err_imag = np.where(err_imag<0, 0, err_imag)
        self.heatmap(err_real, title_, model_name, 'realpartdiff')
        self.heatmap(err_imag, title_, model_name, 'imagpartdiff')
        
    def phase_diff_fig(self, img1, img2, title_='angle difference', threshold=None, model_name=None):
        angle_err = self.phase_diff(img1, img2)
        if threshold is None:
            self.heatmap(angle_err, title_, model_name, 'phasediff')
        else:
            self.heatmap((angle_err<threshold).astype(np.float32), title_, model_name, 'phasebinarydiff')
       
    def error_boxplot(self, pred, truth, OBJ, model_name=None):
        pred = self.focusing(pred)
        truth = self.focusing(truth)
        err = {
            "level": [],
            "maxerror": [],
            'mse':[],
            "type": []
            }
        for i in range(pred.shape[0]):
            level, _ =  OBJ.find_level(i, train=False)
            err_real_dist, err_imag_dist = Difference().complex_diff(pred[i], truth[i])
            err["level"].append(level)
            err["maxerror"].append(np.max(err_real_dist))
            err['mse'].append(np.sum(err_real_dist))
            err["type"].append("real")
            err["level"].append(level)
            err["maxerror"].append(np.max(err_imag_dist))
            err['mse'].append(np.sum(err_imag_dist))
            err["type"].append("imag")
        top10error_p = np.argsort(err['maxerror'])[-5:]
        top10error = top10error_p//2
        for i, ind in enumerate(top10error):
            self.complex_diff_fig(pred[ind], truth[ind], title_=str(err['level'][top10error_p[i]]))
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
        plt.plot(err['mse'][::2])
        plt.plot(err['mse'][1::2])
        plt.legend(['real','imag'])
        self.save_fig(model_name, 'errordistplot')
        plt.show()
            
        # err_real = np.zeros((pred.shape[0],))
        # err_imag = np.zeros((pred.shape[0],))
        # levels = np.zeros((pred.shape[0],))
        # level_err_real = []
        # level_err_imag = []
        # try:
        #     for i in range(pred.shape[0]):
        #         levels[i], _ =  OBJ.find_level(i, train=False)
        #         err_real_dist, err_imag_dist = Difference().complex_diff(pred[i], truth[i])
        #         err_real[i], err_imag[i] = np.max(err_real_dist), np.max(err_imag_dist)
        #     plt.figure()
        #     plt.boxplot([err_real,err_imag], labels=['real', 'imag'])
        #     plt.title('maximum error')
        #     self.save_fig(model_name, 'errorboxplot')
        #     plt.show()
        #     for level in range(1,4):
        #         level_err_real.append(err_real[levels==level])
        #         level_err_imag.append(err_imag[levels==level])
        #     plt.figure()
        #     plt.boxplot(level_err_real)
        #     plt.title('real error')
        #     self.save_fig(model_name, 'realerrorboxplot')
        #     plt.show()
        #     plt.figure()
        #     plt.boxplot(level_err_imag)
        #     plt.title('imag error')
        #     self.save_fig(model_name, 'imagerrorboxplot')
        #     plt.show()
        # except AttributeError as e:
        #     print(e)
            
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
        self.phase_diff_fig(self._pred_img, self._truth_img, 'angle diff' + str(self.ind), threshold, self.model_name)
        
    def show_complex_diff(self, DR=None):
        self.complex_diff_fig(self._pred_img, self._truth_img, 'complex diff' + str(self.ind), DR, self.model_name)      

