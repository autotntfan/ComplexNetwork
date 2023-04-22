# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 18:31:41 2022

@author: benzener
"""
import os
if __name__ == '__main__':
    import sys
    currentpath = os.getcwd()
    addpath = os.path.dirname(os.path.dirname(currentpath))
    if addpath not in sys.path:
        sys.path.append(addpath)
    from baseband.setting import constant
    from baseband.utils.data_utils import normalization, downsampling
    from baseband.utils.info import progressbar, save_info, read_info
    sys.path.remove(addpath)
else:
    from ..setting import constant
    from ..utils.data_utils import normalization, downsampling 
    from ..utils.info import progressbar, save_info, read_info
import numpy as np
from scipy import io

class SaveAllData():
    '''
        Matlab data files end with .mat are read and normalized to [-1,1]. Finally, RF and BB PSFs and speckles are
        saved in numpy format file with shape [N,H,W,C], where C is 1 for RF data or 2 for BB data.
        
        Args:
            load_path: string, where data saves.
            saved_cache_path: string, where to save data.
            normalization: boolean, whether to recitify value in the range of [-1,1]
            k: int, how many kinds of phase aberration. i.e. file name: Data_1_delay_k
            
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
                 load_path=constant.DATAPATH,
                 saved_cache_path=constant.CACHEPATH,
                 normalization=True,
                 check_size=True
                 ):
        self.load_path = load_path
        self.saved_cache_path = saved_cache_path
        self.normalization = normalization
        self.check_size = check_size

    def __call__(self):
        # Check data size
        if self.check_size:
            self.__sanitized()
        # Get data in mat file
        dataname = ['psf_bb.npy', 'psf_rf.npy', 'speckle_bb.npy', 'speckle_rf.npy']
        # Read mat files
        psf_bb, psf_rf, speckle_bb, speckle_rf = self.read_mat_file()
        # Convert complex type array to two-channel real-valued array.
        # Expand dimension to 4D array. The last dimension is channel.
        psf_bb = self._expand_dims(psf_bb) # [N,H,W,2]
        psf_rf = self._expand_dims(psf_rf) # [N,H,W,1]
        speckle_bb = self._expand_dims(speckle_bb) # [N,H,W,2]
        speckle_rf = self._expand_dims(speckle_rf) # [N,H,W,1]
        # Normalize in the range of [-1,1]
        if self.normalization:
            print('Normalizing......')
            psf_bb = normalization(psf_bb)
            psf_rf = normalization(psf_rf)
            speckle_bb = normalization(speckle_bb)
            speckle_rf = normalization(speckle_rf)
        # Save data in a cache file
        for ii, data in enumerate((psf_bb, psf_rf, speckle_bb, speckle_rf)):
            self.save_data(data, dataname[ii])
        # Save data information
        save_info({'k':constant.k,'num_total':constant.DATASIZE[0]}, 'parameters.txt', self.saved_cache_path)

    def read_mat_file(self):
        '''
        Read data from matlab files. Memory requirement at least 8 GB.
        '''
        file_name = os.listdir(self.load_path) # file names of all simulation data 
        psf_bb = np.zeros(constant.DATASIZE, dtype=complex)
        psf_rf = np.zeros(constant.DATASIZE)
        speckle_bb = np.zeros(constant.DATASIZE,dtype=complex)
        speckle_rf = np.zeros(constant.DATASIZE)
        count = 0
        # read psf and speckle in sequence
        for name in file_name:
            if name.endswith('.mat'):
                count = count + 1
                # split the i-th level-j PSF 
                ind = np.array(name.split('.')[0].split('_'))[np.char.isnumeric(name.split('.')[0].split('_'))]
                # convert to index
                ind = constant.k*(int(ind[0])-1) + int(ind[1]) - 1 
                file_path = os.path.join(self.load_path, name)
                data = io.loadmat(file_path)
                psf_bb[ind,:,:] = data.get('psf_bb')
                psf_rf[ind,:,:] = data.get('psf_rf')
                speckle_bb[ind,:,:] = data.get('speckle_bb')
                speckle_rf[ind,:,:] = data.get('speckle_rf')
                progressbar(count, constant.DATASIZE[0], 'Loading data')
            else:
                continue
        return psf_bb, psf_rf, speckle_bb, speckle_rf
    
    def save_data(self, data, name):
        '''
        Note: using 'dictionary' to save large array would become slower after hundreds of iteration.
        Instead building four arrays directly ensures memory would be sufficient and run efficiently.
        
        Arg:
            data: ndarray, what to be saved.
            name: string, what is the saved name.
        '''
        # save as npy file
        print(f'Saving {name} ...')
        if not os.path.exists(self.saved_cache_path):
            os.mkdir(self.saved_cache_path)
        np.save(os.path.join(self.saved_cache_path, name), data)
                      
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
        file_path = os.path.join(self.load_path, validname)
        data = io.loadmat(file_path)
        shape = data.get('psf_bb').shape
        return (num,) + shape
    
    def _expand_dims(self, x):
        '''
        Expand input dimension to [N,H,W,C]. 
        input is real -> [N,H,W,1]
        input is complex -> [N,H,W,2]
        '''
        if x.dtype == complex:
            real = np.expand_dims(x.real, axis=-1)
            imag = np.expand_dims(x.imag, axis=-1)
            z = np.concatenate((real,imag), axis=-1)
            return z.astype(np.float32)
        else:
            real = np.expand_dims(x, axis=-1)
            return real.astype(np.float32)
    
    def __sanitized(self):
        '''
        Automatically check if the data size and path are compatible with them in constant.py
        '''
        file_name = os.listdir(self.load_path) # file names of all simulation data
        if self.load_path == constant.DATAPATH:
            assert self.get_data_size(file_name) == constant.DATASIZE, 'Check the data size'
        else:
            raise ValueError('Unsupported path')
            

class GetData(SaveAllData):
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
                 down_sample_factor=1,
                 num_training=1800,
                 validation_split=0,
                 complex_network=True,
                 forward=False,
                 seed=7414,
                 **kwargs):
        super().__init__(**kwargs)
        self.down_sample_factor = down_sample_factor # downsampling factor along axial direction
        self.num_training = num_training # number of training data + validation data
        self.validation_split = validation_split
        self.complex_network = complex_network # use RF or BB model
        self.forward = forward # whether use forward path
        self.seed = seed # random seed
    
        self.indices = np.arange(constant.DATASIZE[0]) # initialized indices 
        self.dataset = None # a dictionary for training pair
        self._recreate = False
        
    def __call__(self):
        if self.num_training > constant.DATASIZE[0]:
            raise ValueError(f"Dataset only has {constant.DATASIZE[0]} samples but requires {self.num_training} training data")
        print('Trying to check files ... \n ')
        self.__sanitized()
        if self._recreate:
            self.info('w')
            
        # suffle indices
        self.__shuffle_ind()
        # select rf or BB data
        if self.complex_network:
            dtype = ['speckle_bb', 'psf_bb']
        else:
            dtype = ['speckle_rf', 'psf_rf']
        # build dictionary to search saving or saved name
        if self.validation_split:
            self.dataset = {
                dtype[0]:['x_train', 'x_test', 'x_valid'],
                dtype[1]:['y_train', 'y_test', 'y_valid']
                }

        else:
            self.dataset = {
                dtype[0]:['x_train', 'x_test'],
                dtype[1]:['y_train', 'y_test']
                }
        
        x_train, x_test, *x_valid = self.get_data(dtype[0])
        y_train, y_test, *y_valid = self.get_data(dtype[1])

        if self.forward:   
            self.dataset['ideal_psf'] = ['ideal_train', 'ideal_test']   
            ideal_train, ideal_test = self.get_data('ideal_psf')
            return (x_train, y_train), (x_test, y_test), (ideal_train, ideal_test)
        else:
            if self.validation_split:
                return (x_train, y_train), (x_test, y_test), (x_valid[0], y_valid[0])
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
            if self._recreate:
                # recreate cache file
                print(f"{dtype} file is creating ...")
                if dtype in {'ideal_psf'}:
                    if self.complex_network:
                        return self.__slice_dataset('psf_bb', ideal=True)
                    else:
                        return self.__slice_dataset('psf_rf', ideal=True)
                else:
                    return self.__slice_dataset(dtype)
            # not require to recreate file -> read cache file
            else:
                if self.validation_split:
                    return self.__read_cache(self.dataset[dtype][0]), self.__read_cache(self.dataset[dtype][1]), self.__read_cache(self.dataset[dtype][2])
                else:
                    return self.__read_cache(self.dataset[dtype][0]), self.__read_cache(self.dataset[dtype][1])
        except FileNotFoundError:
            print(f"File is not found, trying to recreate {dtype} file")
            if dtype in {'ideal_psf'}:
                if self.complex_network:
                    return self.__slice_dataset('psf_bb', ideal=True)
                else:
                    return self.__slice_dataset('psf_rf', ideal=True)
            else:
                return self.__slice_dataset(dtype)
        
    def __sanitized(self):
        prepar = self.info('r') # if not found -> raise FileNotFoundError -> recreate root file
        nowpar = self.info()
        if len(prepar.keys()) != len(nowpar.keys()):
            # only root file has created -> create sub file
            self._recreate = True
        for key in prepar.keys():
            if prepar[key] != nowpar[key]:
                if key in {'k','num_total', 'saved_cache_path', 'load_path'}:
                    raise ValueError('Root file requires to recreate')
                else:
                    # recreate sub file
                    self._recreate = True
        
    def get_indices(self, key):
        if key == 'train':
            return self.indices[0]
        elif key == 'test':
            return self.indices[1]
        elif key == 'valid':
            return self.indices[2]
        else:
            raise ValueError(f"Input argument can only be 'train', 'test', or 'valid' but get {key}")

    
    def find_level_and_ind(self, ind=None, train=True):
        '''
        Get phase aberration level according to the index. If index is None, this function returns the whole
        indices.
            Args:
                ind: int or None, the i-th data or whole dataset.
                train: boolean, this index is from training data or testing data.
        '''
        level = np.arange(constant.k) + 1 # convert range from [0,k-1] to [1,k].
        level = np.tile(level, constant.DATASIZE[0]//constant.k)


        if train:
            train_indices = self.get_indices('train')
            level_of_train = level[train_indices]
            if ind is None:
                return level_of_train, train_indices
            else:
                return level_of_train[ind], train_indices[ind]
        else:
            test_indices = self.get_indices('test')
            level_of_test = level[test_indices]
            if ind is None:
                return level_of_test, test_indices
            else:
                return level_of_test[ind], test_indices[ind]
        
    def info(self, op=None):
        '''
        Input arguments setting
        '''
        file_name = 'parameters.txt'
        training_data_shape = self.training_data_shape()
        saved_var = {
            'k':constant.k,
            'validation_split':self.validation_split,
            'num_total':constant.DATASIZE[0],
            'num_training':training_data_shape[0],
            'num_testing':constant.DATASIZE[0] - self.num_training,
            'num_valid':self.num_training - training_data_shape[0],
            'down_sample_factor':self.down_sample_factor,
            'complex_network':self.complex_network,
            'forward':self.forward,
            'seed':self.seed,
            'saved_cache_path':self.saved_cache_path,
            'load_path':self.load_path
            }
        if op is None:
            return saved_var
        elif op == 'r':
            return read_info(file_name, self.saved_cache_path)
        elif op == 'w':
            save_info(saved_var, file_name, self.saved_cache_path)
        else:
            raise ValueError("Expected 'r', 'w', and NoneType for reading, writing and getting information.")
            
            
    def __shuffle_ind(self):
        '''
        Randomly suffle data indices.
        '''
        rng = np.random.default_rng(self.seed)
        num_valid = round(self.num_training*self.validation_split)
        train_indices = self.indices[:self.num_training-num_valid]
        test_indices = self.indices[self.num_training:]
        rng.shuffle(train_indices)
        rng.shuffle(test_indices)
        if self.validation_split:
            valid_indices = self.indices[self.num_training-num_valid:self.num_training]
            rng.shuffle(valid_indices)
            self.indices = (train_indices, test_indices, valid_indices)
        else:
            self.indices = (train_indices, test_indices)
    
    def __slice_dataset(self, dtype, ideal=False):
        '''
        Parameters
        ----------
        dtype : str, sliced image type
            It can be 'speckle_bb', 'psf_bb' or 'speckle_rf', 'psf_rf'.
        ideal : boolean, whether ideal PSF is required.

        '''
        path = os.path.join(self.saved_cache_path, dtype+'.npy') # read cache file saved from the parent class.
        # load data saved by Preprocessing(), if the data size is odd, the first column or row is removed.
        data = np.load(path)[:, constant.DATASIZE[1]%2:, constant.DATASIZE[2]%2:, :]
        # decimation
        if self.down_sample_factor:
            data = downsampling(data, self.down_sample_factor, 'axial')
        if ideal:
            if constant.k == 4:
                ideal_psf = data[::4] 
                data = np.repeat(ideal_psf, 4, axis=0) # replicate by 4 times
                assert data.shape[0] == constant.DATASIZE[0]
                assert (data[0] == data[3]).all()
                dtype = 'ideal_psf'
            else:
                raise ValueError(f"Obtain idea PSF only support for k = 4, but get k ={constant.k}")
        train_indices, test_indices, *valid_indices = self.indices # random shuffle
        
        train = data[train_indices]
        self.__save_cache(train, self.dataset[dtype][0]) 
        test = data[test_indices]
        self.__save_cache(test, self.dataset[dtype][1])
        assert train.shape == self.training_data_shape(), 'Training data size is wrong.'
        if self.validation_split:
            valid_indices = valid_indices[0]
            valid = data[valid_indices]
            self.__save_cache(valid, self.dataset[dtype][2])
            return train, test, valid
        else:
            return train, test
    

    def __save_cache(self, x, name):
        '''
        Save training pairs.
        '''
        assert isinstance(name, str)
        print(f'Saving {name} ... ')
        np.save(os.path.join(self.saved_cache_path, name + '.npy'), x)
    
    def __read_cache(self, name):
        assert isinstance(name, str)
        print(f'Reading {name} ... ')
        return np.load(os.path.join(self.saved_cache_path, name + '.npy'))
            
    
    def training_data_shape(self):
        output_shape = (round(self.num_training*(1-self.validation_split)),) + \
            ((constant.DATASIZE[1]-constant.DATASIZE[1]%2)//self.down_sample_factor,) + \
                (constant.DATASIZE[1]-constant.DATASIZE[1]%2,)
        if self.complex_network:
            return output_shape + (2,)
        else:
            return output_shape + (1,)
        
    # def get_ideal(self):
    #     speckle_path = os.path.join(self.DIRECTORY, 'speckle_bb.npy')
    #     psf_path = os.path.join(self.DIRECTORY, 'psf_bb.npy')
    #     speckles = self.__load_file(speckle_path) # read data
    #     psfs = self.__load_file(psf_path) # read data
    #     if self.down_sample_factor:
    #         speckles = downsampling(speckles, self.down_sample_factor, 'axial')
    #         psfs = downsampling(psfs, self.down_sample_factor, 'axial')
    #     if constant.k == 8:
    #         ideal_psf = np.zeros((int(constant.DATASIZE[0]/constant.k),128,256,2), dtype=np.float32)
    #         ideal_speckle = np.zeros((int(constant.DATASIZE[0]/constant.k),128,256,2), dtype=np.float32)
    #         num = 0
    #         for i in range(constant.DATASIZE[0]):
    #             if i == (i//40+1+8*(i//8)):
    #                 ideal_speckle[num] = speckles[i]
    #                 ideal_psf[num] = psfs[i]
    #                 num = num + 1
    #         self.__save_cache(ideal_speckle, 'ideal_speckle.npy')
    #         self.__save_cache(ideal_psf, 'ideal_psf.npy')
    #         return ideal_speckle, ideal_psf
    # def get_ideal(self):
    #     file_name = os.listdir(r'../MATLAB/idealPSF') # file names of all simulation data 
    #     ideal_psf = np.zeros((int(constant.DATASIZE[0]/constant.k),128,256,2), dtype=np.float32)
    #     ideal_speckle = np.zeros((int(constant.DATASIZE[0]/constant.k),128,256,2), dtype=np.float32)
    
    #     # read psf and speckle in sequence
    #     for name in file_name:
    #         if name.endswith('.mat'):
    #             # split the i-th level-j PSF 
    #             ind = np.array(name.split('.')[0].split('_'))[np.char.isnumeric(name.split('.')[0].split('_'))]
    #             # convert to index
    #             if eval(ind[0]) > constant.DATASIZE[0]/constant.k:
    #                 continue
    #             ind = eval(ind[0]) - 1
    #             file_path = os.path.join(r'../MATLAB/idealPSF', name)
    #             data = io.loadmat(file_path)
    #             psf_bb = data.get('psf_bb')
    #             speckle_bb = data.get('speckle_bb')
    #             psf_bb = self._expand_dims(psf_bb[1::2,1::]) # [N,H,W,2]
    #             speckle_bb = self._expand_dims(speckle_bb[1::2,1::]) # [N,H,W,2]
    #             psf_bb = normalization(psf_bb).astype(np.float32)
    #             speckle_bb = normalization(speckle_bb).astype(np.float32)
    #             ideal_psf[ind] = psf_bb
    #             ideal_speckle[ind] = speckle_bb
                
    #         else:
    #             continue
    #     return ideal_speckle, ideal_psf