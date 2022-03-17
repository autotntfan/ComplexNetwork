# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 23:33:49 2022

@author: benzener
"""
import scipy.io
import numpy as np
import os
import h5py

# -------------------- deal with .mat file --------------------
class DealWithRawData():
    
    def __init__(self, file_num=4, path='D:\ComplexDataset\simulation_straight'):
        self.file_num = file_num
        self.path = path
        
    def save_variables(self):
        # get file name
        file_name = os.listdir(self.path)
        file_name = file_name[1:] # remove count_file.py
        file_name = sorted(file_name,key=lambda x:int(x.replace("Data_","")[:-12])) # sort file name in order
        # data buffer
        data_size = (len(file_name)//self.file_num,513,513)
        with h5py.File('dataset.hdf5','w') as f:
            for level in range(self.file_num):
                names = file_name[level::self.file_num]
                g = f.create_group("level" + str(level+1))
                # data buffer
                psf_bb = np.zeros(data_size, dtype=np.complex)
                psf_rf = np.zeros(data_size)
                speckle_bb = np.zeros(data_size,dtype=np.complex)
                speckle_rf = np.zeros(data_size)
                for i, name in enumerate(names):
                    file_path = os.path.join(self.path, name)
                    psf_bb[i,:,:] = self._get_variable(file_path, 'psf_bb')
                    psf_rf[i,:,:] = self._get_variable(file_path, 'psf_rf')
                    speckle_bb[i,:,:] = self._get_variable(file_path, 'speckle_bb')
                    speckle_rf[i,:,:] = self._get_variable(file_path, 'speckle_rf')
                    if i%10 == 0:
                        print(i)
                psf_bb = self._expand_dims(psf_bb)
                speckle_bb = self._expand_dims(speckle_bb)
                g.create_dataset('psf_bb',psf_bb.shape,dtype='float64',data=psf_bb)
                g.create_dataset('psf_rf',psf_rf.shape,dtype='float64',data=psf_rf)
                g.create_dataset('speckle_rf',speckle_rf.shape,dtype='float64',data=speckle_rf)
                g.create_dataset('speckle_bb',speckle_bb.shape,dtype='float64',data=speckle_bb)
        
        
    def _get_variable(self, path, dtype):
        data = scipy.io.loadmat(path)
        if not isinstance(dtype, str):
            raise AttributeError('key word should be a string')
        if dtype not in data.keys():
            raise KeyError('key word not in data')
        return data.get(dtype)
    
    def _expand_dims(self, x):
        real = np.expand_dims(x.real,axis=-1)
        imag = np.expand_dims(x.imag,axis=-1)
        return np.concatenate((real,imag),axis=-1)


    

'''
structure:
    --group:level 1
        dataset : psf_bb
                  psf_rf
                  speckle_rf
                  speckle_bb
    --group:level 2
    
'''
# -------------------- data preprocessing --------------------
class DataPreprocessing():
    
    def __init__(self,
                 factor=2,
                 path='D:\ComplexDataset\dataset.hdf5',
                 normalization=True,
                 forward=True,
                 training_num=300):
        '''
        Parameters
        ----------
        factor : int
            down-sample ratio of sampling rate. The default is 2.
        
        path : str,
            where to read data. The default is 'D:\ComplxDataset\dataset.hdf5'.
            
        normalization : bool,
            normalize complex value. The default is True.
            a + bi -> (a + bi)/|z|, z is modulus
            
        forward : bool, optional
            using foward UNet. The default is True.
            
        training_num : int, 
            number of training data. The default is 300.
            
        valid_rate : float, 
            seperate validation data from training data. The default is 0.2.

        Returns
            (x_train, y_train), (x_test, y_test), (ideal_train, ideal_test)


        '''
        self.factor = factor
        self.path = path
        self.normalization = normalization
        self.forward = forward
        self.training_num = training_num
        self.indices = None
        
    def get_data(self):
        psf = self._read_data(dtype='psf')
        speckle = self._read_data(dtype='speckle')
        if self.training_num > psf.shape[0]:
            raise ValueError(f'training_num {self.training_num} is larger than data size {psf.shape[0]}')
        if self.normalization:
            psf = self._normalize_complex_value(psf)
            speckle = self._normalize_complex_value(speckle)
        self.indices = np.arange(psf.shape[0])
        np.random.shuffle(self.indices) # suffle order
        if self.factor:
            psf = self._reduce_sampling_rate(psf)
            speckle = self._reduce_sampling_rate(speckle)
            
        x_train = speckle[self.indices[:self.training_num]]
        y_train = psf[self.indices[:self.training_num]]
        x_test = speckle[self.indices[self.training_num:]]
        y_test = psf[self.indices[self.training_num:]]
        if self.forward:
            ideal_psf = np.repeat(psf[::4],4,axis=0)
            assert ideal_psf.shape == psf.shape
            ideal_train = ideal_psf[self.indices[:self.training_num]]
            ideal_test = ideal_psf[self.indices[self.training_num:]]
            return (x_train, y_train), (x_test, y_test), (ideal_train, ideal_test)
        else:
            return (x_train, y_train), (x_test, y_test)
        
    def _read_data(self, dtype='psf'):
        data = None
        if dtype == 'psf':
            type_ = 'psf_bb'
        elif dtype == 'speckle':
            type_ = 'speckle_bb'
        else:
            raise KeyError('dtype is invalid')
        with h5py.File(self.path,'r') as f:
            for level in range(1,5):
                key = 'level' + str(level) + '/' + type_
                img = f[key][:].astype(np.float32)[:,1:,1:,:] # original 513*513 -> 512*512
                if data is None:
                    data = img
                else:
                    data = np.vstack((data,img))
                    break
        return data
    
    def _normalize_complex_value(self, x):
        if len(x.shape) == 3:
            x = np.expand_dims(x,axis=0)
        assert len(x.shape) == 4
        channel = x.shape[-1]//2
        real = x[:,:,:,:channel]
        imag = x[:,:,:,channel:]
        modulus = np.sqrt(real**2 + imag**2)
        ratio = np.max(modulus,axis=(1,2,3)).reshape(x.shape[0],1,1,1)
        return np.nan_to_num(x/ratio)
    
    def _reduce_sampling_rate(self, signal):
        return signal[:,::self.factor,::self.factor,:]

      


    
        
        
    