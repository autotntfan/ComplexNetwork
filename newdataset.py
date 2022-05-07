# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 18:31:41 2022

@author: benzener
"""
import os
from scipy import io
import numpy as np

DATA_SIZE = (2000,257,257)

class DataPreprocessing():
    '''
        Save speckle and psf in rf or BB.
        Read from .mat file and save in numpy format.
        
        function:
            save_data: save psf_bb, psf_rf, speckle_bb, speckle_rf in npy file
            
            __expand_dims: since BB data is complex value, divided into two channel to store real
                           and complex value respectively.
                           original (batch, 513, 513) in complex -> (batch, 513, 513, 2) in real
                           
            __normalization: normalize value in the range [-1, 1].
    '''
    
    def __init__(self,
                 path=r'./simulation_data',
                 normalization=True,
                 ):
        self.path = path
        self.normalization = normalization

    def save_data(self):
        '''
        Note: using 'dictionary' to save large array would become slower after hundreds of iteration.
        Hence, here name four array directly ensuring memory would be sufficient and run efficiently.

        '''
        self.__sanitized()     # check data size is compatible with dataset  
        file_name = os.listdir(self.path)
        psf_bb = np.zeros(DATA_SIZE, dtype=complex)
        psf_rf = np.zeros(DATA_SIZE)
        speckle_bb = np.zeros(DATA_SIZE,dtype=complex)
        speckle_rf = np.zeros(DATA_SIZE)
        # read psf and speckle in sequence
        for i, name in enumerate(file_name):
            if name.split('.')[-1] in {'py','png'}:
                continue
            else:
                ind = np.array(name.split('.')[0].split('_'))[np.char.isnumeric(name.split('.')[0].split('_'))]
                ind = 4*(int(ind[0])-1) + int(ind[1]) - 1
                file_path = os.path.join(self.path, name)
                data = io.loadmat(file_path)
                psf_bb[ind,:,:] = data.get('psf_bb')
                psf_rf[ind,:,:] = data.get('psf_rf')
                speckle_bb[ind,:,:] = data.get('speckle_bb')
                speckle_rf[ind,:,:] = data.get('speckle_rf')
            if (i+1)%100 == 0:
                print(i+1)
        # turn complex-valued array to two channel real-valued one.
        psf_bb = self.__expand_dims(psf_bb)
        speckle_bb = self.__expand_dims(speckle_bb)
        if self.normalization:
            print('now normalizing......')
            psf_bb = self.__normalization(psf_bb)
            psf_rf = self.__normalization(psf_rf)
            speckle_bb = self.__normalization(speckle_bb)
            speckle_rf = self.__normalization(speckle_rf)
        
        # save as npy file
        print('now saving......')
        dir_ = r'./parameters'
        if not os.path.exists(dir_):
            os.mkdir(dir_)
        np.save(os.path.join(dir_, 'psf_bb.npy'), psf_bb)    
        np.save(os.path.join(dir_, 'psf_rf.npy'), psf_rf)    
        np.save(os.path.join(dir_, 'speckle_bb.npy'), speckle_bb)    
        np.save(os.path.join(dir_, 'speckle_rf.npy'), speckle_rf)    

        
    def __expand_dims(self, x):
        assert x.dtype == complex
        real = np.expand_dims(x.real,axis=-1)
        imag = np.expand_dims(x.imag,axis=-1)
        return np.concatenate((real,imag),axis=-1)
    
    def __normalization(self, x):
        # data shape only allows (N, H, W, C)
        if x.ndim == 3:
            x = x.reshape(x.shape+(1,))
        assert x.ndim == 4
        if x.shape[-1]%2:
            ratio = np.max(np.abs(x), axis=(1,2,3)).reshape(x.shape[0],1,1,1)
        else:
            channel = x.shape[-1]//2
            real = x[:,:,:,:channel]
            imag = x[:,:,:,channel:]
            modulus = np.sqrt(real**2 + imag**2)
            ratio = np.max(modulus,axis=(1,2,3)).reshape(x.shape[0],1,1,1)
        return np.nan_to_num(x/ratio).astype(np.float32)
    
    def __sanitized(self):
        if self.path == r'./simulation_data':
            assert DATA_SIZE == (2000,257,257)
        elif self.path == r'./simulation_data':
            assert DATA_SIZE == (1600,513,513)
        else:
            raise ValueError('Unsupport path')
            

class GetData():
    # return training, testing data , and even ideal psf
   
    NUM_DATA = DATA_SIZE[0]
    DIRETORY = r'./parameters'
    
    def __init__(self,
                 factor=1,
                 num_training=1800,
                 complex_network=True,
                 forward=True,
                 seed=7414):
        self.factor = factor
        self.num_training = num_training
        self.complex_network = complex_network
        self.forward = forward
        self.seed = seed
    
        self.indices = None
        self.dataset = None
        
    def __call__(self):
        if self.num_training > self.NUM_DATA:
            raise ValueError(f"Dataset only has {self.NUM_DATA} samples but requires {self.num_training} training data")
        # select rf or BB data
        if self.complex_network:
            dtype = ['speckle_bb', 'psf_bb']
        else:
            dtype = ['speckle_rf', 'psf_rf']
        # build dictionary to search saving or saved name
        self.dataset = {
            dtype[0]:['x_train', 'x_test'],
            dtype[1]:['y_train', 'y_test']
            }
 
        self.__shuffle_data()
        
        x_train, x_test = self.get_data(dtype[0])
        y_train, y_test = self.get_data(dtype[1])

        if self.forward:   
            self.dataset['ideal_psf'] = ['ideal_train', 'ideal_test']
            
            ideal_train, ideal_test = self.get_data('ideal_psf')
            return (x_train, y_train), (x_test, y_test), (ideal_train, ideal_test)
        else:
            return (x_train, y_train), (x_test, y_test)
    
    
    def get_data(self, dtype):
        '''
        Parameters
        ----------
        dtype : str, image type
            It can be 'speckle_bb', 'psf_bb', 'ideal_psf' or 'speckle_rf', 'psf_rf', 'ideal_psf'.

        Returns
        -------
        numpy.array: 
            training data, testing data
        '''
        try:
            print('File is loading ......')
            train = self.__read_cache(self.dataset[dtype][0])
            test = self.__read_cache(self.dataset[dtype][1])
            assert train.shape == self.__output_shape()
            return train, test
        except (FileNotFoundError, AssertionError):
            print('File is not found or inaccurate, now creating ......')
            if dtype in {'ideal_psf'}:
                if self.complex_network:
                    return self.__slice('psf_bb', flag=True)
                else:
                    return self.__slice('psf_rf', flag=True)
            else:
                return self.__slice(dtype)
                
    
    def find_indices(self):
        train_indices = self.indices[:self.num_training]
        test_indices = self.indices[self.num_training:]
        return train_indices, test_indices
    
    def find_level(self, N, train=True):
        level = np.arange(4) + 1
        level = np.tile(level, self.NUM_DATA//4)
        train_indices, test_indices = self.find_indices()
        if train:
            level_of_train = level[train_indices]
            return level_of_train[N], train_indices[N]
        else:
            level_of_test = level[test_indices]
            return level_of_test[N], test_indices[N]
            
    def __shuffle_data(self):
        self.indices = np.arange(self.NUM_DATA)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(self.indices) # suffle order

    def __reduce_sampling_rate(self, signal):
        assert signal.ndim == 4
        return signal[:,::self.factor,:,:]
    
    def __slice(self, dtype, flag=False):
        '''
        Parameters
        ----------
        dtype : str, sliced image type
            It can be 'speckle_bb', 'psf_bb' or 'speckle_rf', 'psf_rf'.

        Returns
        -------
        train : TYPE
            DESCRIPTION.
        test : TYPE
            DESCRIPTION.

        '''
        path = os.path.join(self.DIRETORY, dtype+'.npy')
        data = self.__load_file(path)
        # decimation
        if self.factor:
            data = self.__reduce_sampling_rate(data)
        if flag:
            ideal_psf = data[::4]          
            data = np.repeat(ideal_psf, 4, axis=0)
            assert data.shape[0] == self.NUM_DATA
            assert (data[0] == data[3]).all()
            dtype = 'ideal_psf'
        train_indices, test_indices = self.find_indices()
        train = data[train_indices]
        test = data[test_indices]
        self.__save_cache(train, self.dataset[dtype][0])
        self.__save_cache(test, self.dataset[dtype][1])
        return train, test
    
    def __save_cache(self, x, name):
        assert isinstance(name, str)
        np.save(os.path.join(self.DIRETORY, name + '.npy'), x)
        
    def __read_cache(self, name):
        assert isinstance(name, str)
        return np.load(os.path.join(self.DIRETORY, name + '.npy'))
    
    def __load_file(self, path):
        return np.load(path)[:, DATA_SIZE[1]%2:, DATA_SIZE[2]%2:, :]
    
    def __output_shape(self):
        output_shape = (self.num_training,) + \
            ((DATA_SIZE[1]-DATA_SIZE[1]%2)//self.factor,) + \
                (DATA_SIZE[1]-DATA_SIZE[1]%2,)
        if self.complex_network:
            return output_shape + (2,)
        else:
            return output_shape + (1,)
