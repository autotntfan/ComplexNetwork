# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 18:31:41 2022

@author: benzener
"""
import os
import numpy as np
from scipy import io
if __name__ == '__main__':
    import sys
    currentpath = os.getcwd()
    addpath = os.path.dirname(os.path.dirname(currentpath))
    if addpath not in sys.path:
        sys.path.append(addpath)
    from baseband.setting import constant
    from baseband.utils.data_utils import normalization, downsampling, standardization
    from baseband.utils.info import progressbar, save_info, read_info
    sys.path.remove(addpath)
else:
    from ..setting import constant
    from ..utils.data_utils import normalization, downsampling, standardization
    from ..utils.info import progressbar, save_info, read_info

class BaseProcess():
       
    def get_dataset_size(self, path):
        '''
        Automatically calculate the number of files and size of simulation data in the directory.
        Args:
            path: String, path of a directory or a "npy" file. If it is a directory, this function will calculate
            the number of matlab files in the directory (N) and get the speckle size (H,W) saved in the last file. 
            If it is a file, this function will calculate the data size (H,W) saved in this file.
        Returns:
            tuple, shape of data in [N,H,W] or [H,W], where N is the number of files. And the other dimension is
            determined by the saved data
        '''
        if isinstance(path, str) and path.endswith('.npy'):
            data = self.read_npy_file(path)
            return data.shape
        file_name = os.listdir(path) # file names of all simulation data 
        num = 0
        for i, name in enumerate(file_name):
            if name.endswith('.mat'):
                num += 1
                validname = name
            else:
                continue
        file_path = os.path.join(path, validname)
        data = io.loadmat(file_path)
        shape = data.get('speckle_rf').shape
        return (num,) + shape
    
    def expand_dim(self, x):
        '''
        Expand input dimension from [N,H,W] to [N,H,W,C]. 
        input is real -> [N,H,W,1]
        input is complex -> [N,H,W,2]
        '''
        return np.expand_dims(x, axis=-1)
 
        
    def save_npy_file(self, data, path, file_name=None):
        '''
        Save data as npy format.       
        Arg:
            data: ndarray, what to be saved.
            path: String, a directory if file_name is not None, otherwise save file regarding to this path.
            file_name: string, what is the saved name.
        '''
        # save as npy file
        if file_name is not None:
            if not os.path.exists(path):
                os.mkdir(path)
            path = os.path.join(path,file_name)
        if not path.endswith('.npy'):
            path = path + '.npy'
        np.save(path, data)
        
    def read_npy_file(self, path, file_name=None):
        '''
        Read npy format file
        Args:
            path: String, a directory if file_name is not None. Otherwise, save file regarding to this path.
            file_name: string, what is the saved name.
        '''
        if file_name is not None:
            path = os.path.join(path, file_name)
        if not path.endswith('.npy'):
            path = path + '.npy'
        return np.load(path)           
    
    def check_datasize(self, path):
        '''
        Automatically check if the data size and path are compatible with which in constant.py
        '''
        if path == constant.DATAPATH:
            msg = f"Get data size is {self.get_dataset_size(path)}, but it is required to be {constant.DATASIZE} in constant"
            assert self.get_dataset_size(path) == constant.DATASIZE, msg
        else:
            raise ValueError('Path of dataset is wrong.')
            
class SaveAllData(BaseProcess):
    '''
        Matlab data files end with .mat are read and normalized to [-1,1]. Finally, RF and BB PSFs and speckles are
        saved in numpy format file with shape [N,H,W,C], where C is 1 for RF data or 2 for BB data.
        
        Args:
            k: int, how many kinds of phase aberration. i.e. file name: Data_1_delay_k
            dataset_path: string, where data saves.
            cache_path: string, where to save data.
            normalize: boolean, whether to normalize value in the range of [-1,1]
            standardize: boolean, whether to standardize value. Data will have zero mean and unit standard deviation.
        Example:
            func = SaveAllData(
                k=4,
                dataset_path=r'./MatlabCheck/simulation_data2',
                cache_path=r'./parameters',
                normalize=True,
                standardize=False
                )
            func()
            
    '''
    
    def __init__(self,
                 num_required_data=None,
                 k=4,
                 dataset_path=r'./MatlabCheck/simulation_data2',
                 cache_path=r'./parameters',
                 normalize=True,
                 standardize=False):
        self.num_required_data = num_required_data # how many data would be used
        self.k = k # k-kind of phase aberration or speed of sound
        self.dataset_path = dataset_path # path of simulation data
        self.cache_path = cache_path # path of saved data in npy format
        self.normalize = normalize # whether to normalize data
        self.standardize = standardize # whether to standardize data

    def __call__(self):
        file_name = os.listdir(self.dataset_path) # file names of all simulation data 
        # Check data size
        # self.check_datasize(self.dataset_path)
        data_size = self.get_dataset_size(self.dataset_path)
        if self.num_required_data is not None:
            data_size = list(data_size)
            data_size[0] = self.num_required_data
            data_size = tuple(data_size)
        psf_bb = np.zeros(data_size, dtype=np.complex64)
        psf_rf = np.zeros(data_size, dtype=np.float32)
        speckle_bb = np.zeros(data_size, dtype=np.complex64)
        speckle_rf = np.zeros(data_size, dtype=np.float32)
        count = 0
        # read psf and speckle in sequence
        for name in file_name:
            # contain mat and figure files
            if name.endswith('.mat'):
                # split the i-th level-j PSF 
                ind = np.array(name.split('.')[0].split('_'))[np.char.isnumeric(name.split('.')[0].split('_'))]
                # convert to index
                ind = self.k*(int(ind[0])-1) + int(ind[1]) - 1 
                # Note: because name in file_name is not in the order what we want
                if ind >= self.num_required_data:
                    continue
                count = count + 1
                file_path = os.path.join(self.dataset_path, name)
                data = io.loadmat(file_path)
                psf_bb[ind,:,:] = data.get('psf_bb')
                psf_rf[ind,:,:] = data.get('psf_rf')
                speckle_bb[ind,:,:] = data.get('speckle_bb')
                speckle_rf[ind,:,:] = data.get('speckle_rf')
                progressbar(count, data_size[0], 'Loading ...')
            else:
                # pass figure files
                continue
        # Convert complex type array to two-channel real-valued array.
        # Expand dimension to 4D array. The last dimension is channel.
        psf_bb = self.expand_dim(psf_bb) # [N,H,W,2]
        psf_rf = self.expand_dim(psf_rf) # [N,H,W,1]
        speckle_bb = self.expand_dim(speckle_bb) # [N,H,W,2]
        speckle_rf = self.expand_dim(speckle_rf) # [N,H,W,1]
        # Normalize in the range of [-1,1]
        if self.normalize:
            print('Normalizing......')
            psf_bb = normalization(psf_bb)
            psf_rf = normalization(psf_rf)
            speckle_bb = normalization(speckle_bb)
            speckle_rf = normalization(speckle_rf)
        # Standardize to zero mean and unit std
        if self.standardize:
            print('Standardizing......')
            psf_bb = standardization(psf_bb)
            psf_rf = standardization(psf_rf)
            speckle_bb = standardization(speckle_bb)
            speckle_rf = standardization(speckle_rf)
        data_list = ['psf_bb.npy', 'psf_rf.npy', 'speckle_bb.npy', 'speckle_rf.npy']
        # Save data in cache files
        print('Saving ...')
        for ii, data in enumerate((psf_bb, psf_rf, speckle_bb, speckle_rf)):
            self.save_npy_file(data, self.cache_path, data_list[ii])
        # Save data information
        save_info({'k':self.k,'num_total':data_size[0]}, 'parameters.txt', self.cache_path) 
            

class GetData(SaveAllData):
    '''
    Divide data into training, testing, and (validation) sets.
    Note: Because forward network has almost no use. It would give rise to some errors when returninng data.
    Args:
        down_sample_factor: Int, downsampling factor along axial direction. For baseband data, due to the lower sampling
        rate, this factor can be set as 2. It increase the interval along the depth direction. For RF data, it should be 1.
        num_training: Int, number of training data + validation data
        validation_split: float, from 0 to 1, ratio of validation set in num_training
        complex_data: boolean, return radio-frequency or baseband data
        forward: boolean, also return ideal PSFs
        seed: int, random seed
        shuffle_method: 'normal' or 'mix', suffle index method, if normal, it ensures the training and testing data are
        independent. if mix, the training and testing data may has same properties such as frequency or scatterer, which
        is determined by the input dataset.
    Example:
        Get my training pair of complex network
        func = GetData(
            down_sample_factor=2,
            num_training=1800,
            validation_split=0.05,
            complex_data=True,
            forward=False,
            seed=7414,
            shuffle_method='normal',
            )
        (x_train, y_train), (x_test, y_test), (x_val, y_val) = func()
        
        If you want to test in RF mode and without validation set
        func = GetData(
            down_sample_factor=1,
            num_training=1800,
            validation_split=0,
            complex_data=False,
            forward=False,
            seed=7414,
            shuffle_method='normal',
            )
        (x_train, y_train), (x_test, y_test), (x_val, y_val) = func()
        
        If you want to model learn the scatterer and frequency property
        func = GetData(
            down_sample_factor=2,
            num_training=1800,
            validation_split=0.05,
            complex_data=True,
            forward=False,
            seed=7414,
            shuffle_method='mix',
            )
        (x_train, y_train), (x_test, y_test), (x_val, y_val) = func()
        
    '''
    def __init__(self,
                 down_sample_factor=2,
                 num_training=1800,
                 validation_split=0.05,
                 complex_data=True,
                 forward=False,
                 seed=7414,
                 shuffle_method='normal',
                 **kwargs):
        super().__init__(**kwargs)
        self.down_sample_factor = down_sample_factor # downsampling factor along axial direction
        self.num_training = num_training # number of training data + validation data
        self.validation_split = validation_split # ratio of validation set in num_training
        self.complex_data = complex_data # return radio-frequency or baseband data
        self.forward = forward # whether use forward path
        self.seed = seed # random seed
        if shuffle_method not in {'normal', 'mix'}:
            raise ValueError(f"Shuffle method can be 'normal' or 'mix' but get {shuffle_method}")
        self.shuffle_method = shuffle_method
    
        self.shuffle = False
        
    def __call__(self):
        if self.num_training > constant.DATASIZE[0]:
            raise ValueError(f"Dataset only has {constant.DATASIZE[0]} samples but requires {self.num_training} training data")

        # select rf or BB data
        if self.complex_data:
            data_list = ['speckle_bb', 'psf_bb']
        else:
            data_list = ['speckle_rf', 'psf_rf']
        try:
            # check cache exists
            x = self.read_npy_file(self.cache_path, data_list[0])
            y = self.read_npy_file(self.cache_path, data_list[1])
            x = x[:,1:,1:,:]
            y = y[:,1:,1:,:]
            self.dataset_size = {
                'num_total':len(x),
                'num_testing':len(x) - self.num_training
                }
            prepar = self.info('r') # previous saved parameters in parameter.txt
        except FileNotFoundError:
            print('Root file not found')
            # if root cache hasn't been create, need to call SaveAllData
            raise 
        except Exception as e:
            print(e)
        try:
            nowpar = self.info() # check parameter setting is matching
            if len(prepar.keys()) != len(nowpar.keys()):
                # recreate training/testing chache file
                raise AssertionError('Parameter file is not compatible')
            for key in prepar.keys():
                if prepar[key] != nowpar[key]:
                    if key in {'k','num_total', 'cache_path', 'load_path'}:
                        raise ValueError
                    else:
                        # recreate training/testing chache file
                        raise AssertionError
            return self.read_cache_file()
            
        except FileNotFoundError:
            print('Cache file not found')
            # create training/testing chache file
            pass
        except AssertionError:
            print('Parameter file is not compatible, cache file need to be recreated')
            # create training/testing chache file
            pass
        except ValueError:
            # recreate root chache file
            print('Root file requires to recreate')
            raise
        except Exception as e:
            print(e)
        return self.create_cache_file(x, y)
    
    def read_cache_file(self):
        print('Reading cache ...')
        x_train = self.read_npy_file(self.cache_path, 'x_train')
        x_test = self.read_npy_file(self.cache_path, 'x_test')
        y_train = self.read_npy_file(self.cache_path, 'y_train')
        y_test = self.read_npy_file(self.cache_path, 'y_test')

        if self.validation_split > 0:
            x_val = self.read_npy_file(self.cache_path, 'x_val')
            y_val = self.read_npy_file(self.cache_path, 'y_val')
            self.indices = np.arange(len(x_train)+len(x_test)+len(x_val))
            if self.forward:
                ideal_train = self.read_npy_file(self.cache_path, 'ideal_train')
                ideal_test = self.read_npy_file(self.cache_path, 'ideal_test')
                ideal_val = self.read_npy_file(self.cache_path, 'ideal_val')
                return (x_train, y_train), (x_test, y_test), (x_val, y_val), (ideal_train, ideal_test, ideal_val)
            else:
                return (x_train, y_train), (x_test, y_test), (x_val, y_val)
        else:
            self.indices = np.arange(len(x_train)+len(x_test))
            if self.forward:
                ideal_train = self.read_npy_file(self.cache_path, 'ideal_train')
                ideal_test = self.read_npy_file(self.cache_path, 'ideal_test')
                return (x_train, y_train), (x_test, y_test), (ideal_train, ideal_test)
            else:
                return (x_train, y_train), (x_test, y_test)

    def create_cache_file(self, x, y):
        print('Creating cache file ...')
        x = downsampling(x, self.down_sample_factor, 'axial')
        y = downsampling(y, self.down_sample_factor, 'axial')
        if self.forward:
            if self.k == 4:
                ideal_psfs = y[::4]
                ideal_psfs = np.repeat(ideal_psfs, 4, axis=0) # replicate by 4 times
            else:
                raise ValueError(f"Obtain idea PSF only support for k = 4, but get k ={self.k}")
        self.indices = np.arange(len(x))
        self.info('w')
        x_train = x[self.get_ind('train')]
        y_train = y[self.get_ind('train')]
        x_test = x[self.get_ind('test')]
        y_test = y[self.get_ind('test')]
        self.save_npy_file(x_train, self.cache_path, 'x_train')
        self.save_npy_file(y_train, self.cache_path, 'y_train')
        self.save_npy_file(x_test, self.cache_path, 'x_test')
        self.save_npy_file(y_test, self.cache_path, 'y_test')
        if self.validation_split > 0:
            x_val = x[self.get_ind('val')]
            y_val = y[self.get_ind('val')]
            self.save_npy_file(x_val, self.cache_path, 'x_val')
            self.save_npy_file(y_val, self.cache_path, 'y_val')
            if self.forward:
                ideal_train = ideal_psfs[self.get_ind('train')]
                ideal_test = ideal_psfs[self.get_ind('test')]
                ideal_val = ideal_psfs[self.get_ind('val')]
                self.save_npy_file(ideal_train, self.cache_path, 'ideal_train')
                self.save_npy_file(ideal_test, self.cache_path, 'ideal_test')
                self.save_npy_file(ideal_val, self.cache_path, 'ideal_val')
                return (x_train, y_train), (x_test, y_test), (x_val, y_val), (ideal_train, ideal_test, ideal_val)
            else:
                return (x_train, y_train), (x_test, y_test), (x_val, y_val)
        else:
            if self.forward:
                ideal_train = ideal_psfs[self.get_ind('train')]
                ideal_test = ideal_psfs[self.get_ind('test')]
                self.save_npy_file(ideal_train, self.cache_path, 'ideal_train')
                self.save_npy_file(ideal_test, self.cache_path, 'ideal_test')
                return (x_train, y_train), (x_test, y_test), (ideal_train, ideal_test)
            else:
                return (x_train, y_train), (x_test, y_test)
        
    def shuffle_ind(self):
        '''
        Randomly suffle data indices.
        '''
        rng = np.random.default_rng(self.seed)
        if self.shuffle_method == 'mix':
            # mix the training and testing data, i.e. model potentially learn the testing data
            rng.shuffle(self.indices)
        test_indices = self.indices[self.num_training:]
        if self.validation_split > 0:
            # num_training = # of training data + # of validation data
            # |training data|validation data|testing data|
            num_valid = round(self.num_training*self.validation_split)
            train_indices = self.indices[:self.num_training-num_valid]
            valid_indices = self.indices[self.num_training-num_valid:self.num_training]
        else:
            train_indices = self.indices[:self.num_training]
        
        if self.shuffle_method == 'normal':
            # You should use this way
            rng.shuffle(train_indices)
            rng.shuffle(test_indices)
            if self.validation_split > 0:
                rng.shuffle(valid_indices)
        if self.validation_split > 0:
            self.indices = (train_indices, test_indices, valid_indices)
        else:
            self.indices = (train_indices, test_indices)
        self.shuffle = True
        
    def get_ind(self, key):
        if not self.shuffle:
            self.shuffle_ind()
            self.shuffle = True 
        if key == 'train':
            return self.indices[0]
        elif key == 'test':
            return self.indices[1]
        elif key == 'val':
            return self.indices[2]
        else:
            raise ValueError(f"Input argument can only be 'train', 'test', or 'val' but get {key}")

    def find_level_and_ind(self, key, ind=None):
        '''
        Get phase aberration level according to the index. If index is None, this function returns the whole
        indices.
            Args:
                key: 'train','test', or 'val'. Want to get training, testing or validation's indices.
                ind: int or None, the i-th data or whole dataset.

        '''
        if key not in {'train', 'test','val'}:
            raise ValueError(f"Key should be 'train', 'test' or 'val', but get {key}")
        level = np.arange(self.k) + 1 # convert range from [0,k-1] to [1,k].
        level = np.tile(level, constant.DATASIZE[0]//self.k)
        indices = self.get_ind(key)
        if ind is None:
            return level[indices], indices
        else:
            indth_level = level[indices]
            return indth_level[ind], indices[ind]

        
    def info(self, op=None):
        '''
        Input arguments setting
        Arg:
            op:None,'r',or 'w'. if None, return the whole dictionary of parameters. if 'r', read the saved
            file parameters.txt. If 'w', write parameters into parameters.txt.
        '''
        file_name = 'parameters.txt'
        saved_var = {
            'k':self.k,
            'validation_split':self.validation_split,
            'num_total':self.dataset_size['num_total'],
            'num_training':round(self.num_training*(1-self.validation_split)),
            'num_testing':self.dataset_size['num_testing'],
            'num_valid':round(self.num_training*self.validation_split),
            'down_sample_factor':self.down_sample_factor,
            'complex_data':self.complex_data,
            'forward':self.forward,
            'seed':self.seed,
            'cache_path':self.cache_path,
            'load_path':self.dataset_path,
            'shuffle_method':self.shuffle_method
            }
        if op is None:
            return saved_var
        elif op == 'r':
            return read_info(file_name, self.cache_path)
        elif op == 'w':
            save_info(saved_var, file_name, self.cache_path)
        else:
            raise ValueError("Expected 'r', 'w', and NoneType for reading, writing and getting information.")

        
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