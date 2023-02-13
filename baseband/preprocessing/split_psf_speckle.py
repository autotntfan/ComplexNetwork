# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 18:31:41 2022

@author: benzener
"""
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
    from baseband.utils.data_utils import normalization, downsampling    
    sys.path.remove(addpath)
else:
    from ..setting import constant
    from ..utils.data_utils import normalization, downsampling 
import numpy as np
from scipy import io

class DataPreprocessing():
    '''
        Matlab data files end with .mat are read and normalized to [-1,1]. Finally, RF and BB PSFs and speckles are
        saved in numpy format file with shape [N,H,W,C], where C is 1 for RF data or 2 for BB data.

        function:
            save_data: main function, save psf_bb, psf_rf, speckle_bb, speckle_rf in npy files.
                        e.g. process = DataPreprocessing(path, normalization=True)
                             process.save_data()
            
            
            __expand_dims: BB data derived from Matlab file is complex dtype. For convenience, it's divided into two 
                           channels to store real and imaginary parts respectively.
                           original (N, 513, 513) in complex data type -> (N, 513, 513, 2) in real data type
                           e.g.
                           [4+3i,2-3i] -> [[4,3],[2,-3]]
                           
            __normalization: normalize value in the range [-1, 1].
    '''
    
    def __init__(self,
                 path=constant.DATAPATH,
                 normalization=True,
                 ):
        self.path = path
        self.normalization = normalization

    def save_data(self):
        '''
        Note: using 'dictionary' to save large array would become slower after hundreds of iteration.
        Hence, here building four arrays directly ensures memory would be sufficient and run efficiently.

        '''
        self.__sanitized()
        file_name = os.listdir(self.path) # file names of all simulation data 
        psf_bb = np.zeros(constant.DATASIZE, dtype=complex)
        psf_rf = np.zeros(constant.DATASIZE)
        speckle_bb = np.zeros(constant.DATASIZE,dtype=complex)
        speckle_rf = np.zeros(constant.DATASIZE)
        count = 0
        # read psf and speckle in sequence
        print('Loading ... \n')
        for name in file_name:
            if name.endswith('.mat'):
                count = count + 1
                if count%100 == 0:
                    print(f'Now processing ... >> {count}/{constant.DATASIZE[0]} \n')
                # split the i-th level-j PSF 
                ind = np.array(name.split('.')[0].split('_'))[np.char.isnumeric(name.split('.')[0].split('_'))]
                # convert to index
                ind = 4*(int(ind[0])-1) + int(ind[1]) - 1 
                file_path = os.path.join(self.path, name)
                data = io.loadmat(file_path)
                psf_bb[ind,:,:] = data.get('psf_bb')
                psf_rf[ind,:,:] = data.get('psf_rf')
                speckle_bb[ind,:,:] = data.get('speckle_bb')
                speckle_rf[ind,:,:] = data.get('speckle_rf')
            else:
                continue
        # turn complex type array to two-channel real-valued array.
        psf_bb = self.__expand_dims(psf_bb) # [N,H,W,2]
        psf_rf = self.__expand_dims(psf_rf) # [N,H,W,1]
        speckle_bb = self.__expand_dims(speckle_bb) # [N,H,W,2]
        speckle_rf = self.__expand_dims(speckle_rf) # [N,H,W,1]
        if self.normalization:
            print('Normalizing......')
            psf_bb = self.__normalization(psf_bb)
            psf_rf = self.__normalization(psf_rf)
            speckle_bb = self.__normalization(speckle_bb)
            speckle_rf = self.__normalization(speckle_rf)
        # save as npy file
        print('Saving......')
        if not os.path.exists(constant.CACHEPATH):
            os.mkdir(constant.CACHEPATH)
        np.save(os.path.join(constant.CACHEPATH, 'psf_bb.npy'), psf_bb)    
        np.save(os.path.join(constant.CACHEPATH, 'psf_rf.npy'), psf_rf)    
        np.save(os.path.join(constant.CACHEPATH, 'speckle_bb.npy'), speckle_bb)    
        np.save(os.path.join(constant.CACHEPATH, 'speckle_rf.npy'), speckle_rf)    

        
    def __expand_dims(self, x):
        '''
        Expand input dimension to [N,H,W,C]. 
        input is real -> [N,H,W,1]
        input is complex -> [N,H,W,2]
        '''
        if x.dtype == complex:
            real = np.expand_dims(x.real, axis=-1)
            imag = np.expand_dims(x.imag, axis=-1)
            return np.concatenate((real,imag), axis=-1)
        else:
            return np.expand_dims(x, axis=-1)
            
    
    def __normalization(self, x):
        '''
        Normalize input to [-1,1]
        '''
        return normalization(x).astype(np.float32)

    
    def __sanitized(self):
        '''
        Automatically check if the data size and path are compatible with them in constant.py
        '''
        file_name = os.listdir(self.path) # file names of all simulation data
        if self.path == constant.DATAPATH:
            assert self.get_data_size(file_name) == constant.DATASIZE, 'Check the data size'
        else:
            raise ValueError('Unsupported path')
            
    def get_data_size(self, file_name):
        '''
        Automatically calculate the number of files in the directory.
        '''
        num = 0
        for i, name in enumerate(file_name):
            if name.endswith('.mat'):
                num += 1
                validname = name
            else:
                continue
        file_path = os.path.join(self.path, validname)
        data = io.loadmat(file_path)
        shape = data.get('psf_bb').shape
        return (num,) + shape
            

class GetData(DataPreprocessing):
    '''
    Divide data into training and testing sets.
        function:
            __call__: main function, return training pair. If forward is true, it also returns ideal psf sets.
                     e.g. dataset = GetData(forward=False)
                          (x_train, y_train), (x_test, y_test) = dataset()
                     e.g. dataset = GetData(forward=True)
                          (x_train, y_train), (x_test, y_test), (ideal_train, ideal_test) = dataset()
            get_data: return which data according to the input argument 'dtype'
    '''
    def __init__(self,
                 factor=1,
                 num_training=1800,
                 complex_network=True,
                 forward=True,
                 seed=7414,
                 DIRETORY=constant.CACHEPATH,
                 **kwargs):
        super().__init__(**kwargs)
        self.factor = factor # downsampling factor
        self.num_training = num_training # number of training data
        self.complex_network = complex_network # use RF or BB model
        self.forward = forward # whether use forward path
        self.seed = seed # random seed
        self.DIRETORY  = DIRETORY # path of splitted data from the previous class
    
        self.indices = None # indices after random shuffle
        self.dataset = None # a dictionary for training pair

        
    def __call__(self):
        if self.num_training > constant.DATASIZE[0]:
            raise ValueError(f"Dataset only has {constant.DATASIZE[0]} samples but requires {self.num_training} training data")
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
        print(f'{dtype} file is creating ...')
        if dtype in {'ideal_psf'}:
            if self.complex_network:
                return self.__slice('psf_bb', ideal=True)
            else:
                return self.__slice('psf_rf', ideal=True)
        else:
            return self.__slice(dtype)
    
    def find_indices(self):
        train_indices = self.indices[:self.num_training]
        test_indices = self.indices[self.num_training:]
        return train_indices, test_indices
    
    def find_level(self, ind=None, train=True):
        '''
        Get phase aberration level according to the index. If index is None, this function returns the whole
        indices.
            Args:
                ind: int or None, the i-th data or whole dataset.
                train: boolean, this index is from training data or testing data.
        '''
        level = np.arange(4) + 1
        level = np.tile(level, constant.DATASIZE[0]//4)
        train_indices, test_indices = self.find_indices()
        if train:
            level_of_train = level[train_indices]
            if ind is None:
                return level_of_train, train_indices
            else:
                return level_of_train[ind], train_indices[ind]
        else:
            level_of_test = level[test_indices]
            if ind is None:
                return level_of_test, test_indices
            else:
                return level_of_test[ind], test_indices[ind]
        
    def info(self):
        '''
        Input arguments setting
        '''
        saved_var = {
            'factor':self.factor,
            'num_training':self.num_training,
            'seed':self.seed
            }
        return saved_var
            
    def __shuffle_data(self):
        '''
        Randomly suffle data indices.
        '''
        self.indices = np.arange(constant.DATASIZE[0])
        rng = np.random.default_rng(self.seed)
        rng.shuffle(self.indices) # suffle order

    def __reduce_sampling_rate(self, signal):
        assert signal.ndim == 4
        return signal[:,::self.factor,:,:]
    
    def __slice(self, dtype, ideal=False):
        '''
        Parameters
        ----------
        dtype : str, sliced image type
            It can be 'speckle_bb', 'psf_bb' or 'speckle_rf', 'psf_rf'.
        ideal : boolean, whether ideal PSF is required.

        '''
        path = os.path.join(self.DIRETORY, dtype+'.npy')
        data = self.__load_file(path) # read data
        # decimation
        if self.factor:
            data = downsampling(data, self.factor, 'axial')
        if ideal:
            ideal_psf = data[::4] 
            data = np.repeat(ideal_psf, 4, axis=0) # replicate by 4 times
            assert data.shape[0] == constant.DATASIZE[0]
            assert (data[0] == data[3]).all()
            dtype = 'ideal_psf'
        train_indices, test_indices = self.find_indices() # random shuffle
        train = data[train_indices]
        test = data[test_indices]
        assert train.shape == self.__output_shape(), 'Training data size is wrong.'
        self.__save_cache(train, self.dataset[dtype][0])
        self.__save_cache(test, self.dataset[dtype][1])
        return train, test
    
    def __save_cache(self, x, name):
        '''
        Save training pairs.
        '''
        assert isinstance(name, str)
        np.save(os.path.join(self.DIRETORY, name + '.npy'), x)
    
    def __load_file(self, path):
        '''
        load data saved by Preprocessing(), if the data size is odd, the first column or row is removed.
        '''
        return np.load(path)[:, constant.DATASIZE[1]%2:, constant.DATASIZE[2]%2:, :]
    
    def __output_shape(self):
        output_shape = (self.num_training,) + \
            ((constant.DATASIZE[1]-constant.DATASIZE[1]%2)//self.factor,) + \
                (constant.DATASIZE[1]-constant.DATASIZE[1]%2,)
        if self.complex_network:
            return output_shape + (2,)
        else:
            return output_shape + (1,)
