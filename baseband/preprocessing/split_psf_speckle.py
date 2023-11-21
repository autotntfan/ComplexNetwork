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

class GetData(BaseProcess):
    '''
    Divide data into training, testing, and (validation) sets.
    Note: Because forward network has almost no used long time. It would give rise to some errors when returninng data.
    Args:
        dataset_path: string, where data saved.
        cache_path: string, where to save data.
        num_required_data: int, how many data is used.
        num_training: int, how many data is set as training data.
        validation_split: float, from 0 to 1, ratio of validation set in num_training
        down_sample_factor: Int, downsampling factor along axial direction. For baseband data, due to the lower sampling
        rate, this factor can be set as 2. It increase the interval along the depth direction. For RF data, it should be 1.
        num_training: Int, number of training data + validation data
        
        complex_data: boolean, return radio-frequency (False) or baseband data (True)
        k: int, how many kinds of phase aberration. i.e. file name: Data_1_delay_k
        seed: int, random seed
        shuffle_method: 'normal' or 'mix', suffle index method, if normal, it ensures the training and testing data are
        independent. if mix, the training and testing data may has same properties such as frequency or scatterer, which
        is determined by the input dataset. The `mix` method is shuffle the whole indices firstly and then seperate the
        shuffled indices into training and testing datset, so the similar properties of data would be in the training 
        and testing dataset concurrently. The `normal` method seperates the indices into training and testing datset firstly 
        and then shuffles them respectively, so the similar properties of data must be in the same set either training 
        or testing dataset, individually.
        normalize: boolean, whether to normalize value in the range of [-1,1]. For complex value, it limits the modulus in
        the range instead of real or imaginary part.
        standardize: boolean, whether to standardize value. Data will have zero mean and unit standard deviation.
    Example:
        Get my training pair of complex network (default)
        func = GetData(
            dataset_path=r'./simulation_data_10000',
            cache_path=r'./parameters',
            num_required_data=2000,
            num_training=1800,
            validation_split=0.05,
            down_sample_factor=2,
            complex_data=True,
            forward=False,
            k=4,
            seed=7414,
            shuffle_method='normal',
            normalize=True,
            standardize=False
            )
        (x_train, y_train), (x_test, y_test), (x_val, y_val) = func()
        
        If you want to test in RF mode and without validation set
        func = GetData(
            down_sample_factor=1,
            validation_split=0,
            complex_data=False
            )
        (x_train, y_train), (x_test, y_test) = func()
        
        If you want to model learn the scatterer and frequency properties
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
                 dataset_path=r'./MatlabCheck/simulation_data_10000',
                 cache_path=r'./parameters',
                 num_required_data=None,
                 num_training=1800,
                 validation_split=0.05,
                 down_sample_factor=2,
                 complex_data=True,
                 forward=False,
                 k=4,
                 seed=7414,
                 shuffle_method='normal',
                 normalize=True,
                 standardize=False):
        super().__init__()
        self.dataset_path = dataset_path # path of simulation data
        self.cache_path = cache_path # path of saved data in npy format
        self.num_required_data = num_required_data # how many data would be used
        self.num_training = num_training # number of training data + validation data
        self.validation_split = validation_split # ratio of validation set in num_training
        self.down_sample_factor = down_sample_factor # downsampling factor along axial direction
        self.complex_data = complex_data # return radio-frequency or baseband data
        self.k = k # k-kind of phase aberration or speed of sound
        self.seed = seed # random seed
        self.shuffle_method = shuffle_method
        self.normalize = normalize # whether to normalize data
        self.standardize = standardize # whether to standardize data

        self.shuffle = False
        self._output = self._check()
        
    def __call__(self):
        return self._output
    
    def _check(self):
        if self.shuffle_method not in {'normal', 'mix'}:
            raise ValueError(f"Shuffle method can be 'normal' or 'mix' but get {self.shuffle_method}")
        if self.num_required_data is None:
            self.num_required_data = constant.DATASIZE[0]
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
            assert x.shape == y.shape
            last_ind_H = -1 if x.shape[1]%2 else None # -> remove the last pixel instead of the first pixel due to `depth`
            last_ind_W = -1 if x.shape[2]%2 else None
            x = x[:,:last_ind_H,:last_ind_W,:] 
            y = y[:,:last_ind_H,:last_ind_W,:]
            self.shape = self._output_shape(x)
            self.dataset_size = {
                'num_total':len(x),
                'num_testing':len(x) - self.num_training
                }
            if self.num_required_data != len(x):
                raise AssertionError
            prepar = self.info('r') # previous saved parameters in parameter.txt
        except FileNotFoundError:
            print('Root file not found')
            # if root cache hasn't been create, need to call SaveAllData
            self.save_dataset()
            return self._check()
        except AssertionError:
            print('Number of required data is change, recreating root file.')
            self.save_dataset()
            return self._check()
        except Exception as e:
            print(e)
            raise
        try:
            nowpar = self.info() # check parameter setting is matching
            if len(prepar.keys()) != len(nowpar.keys()):
                # recreate training/testing chache file
                raise AssertionError
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
            print('Parameters in cache file are not compatible, cache file needs to be recreated')
            # create training/testing chache file
            pass
        except ValueError:
            # recreate root chache file
            print('Root file requires to be recreated')
            self.save_dataset()
            return self._check()
        except Exception as e:
            print(e)
        return self.create_cache_file(x, y)
    
    def save_dataset(self):
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
        save_info({'k':self.k,'num_total':data_size[0],'load_path':self.dataset_path,'cache_path':self.cache_path},'parameters.txt', self.cache_path) 

    
    def _output_shape(self, x):
        return (int((x.shape[1] - x.shape[1]%2)/self.down_sample_factor), int(x.shape[2] - x.shape[2]%2))
    
    def read_cache_file(self):
        '''
        Read training.testing.validation chache files from `cache_path`.
        '''
        print('Reading cache ...')
        x_train = self.read_npy_file(self.cache_path, 'x_train')
        x_test = self.read_npy_file(self.cache_path, 'x_test')
        y_train = self.read_npy_file(self.cache_path, 'y_train')
        y_test = self.read_npy_file(self.cache_path, 'y_test')

        if self.validation_split > 0:
            x_val = self.read_npy_file(self.cache_path, 'x_val')
            y_val = self.read_npy_file(self.cache_path, 'y_val')
            self.indices = np.arange(len(x_train)+len(x_test)+len(x_val))
            return (x_train, y_train), (x_test, y_test), (x_val, y_val)
        else:
            self.indices = np.arange(len(x_train)+len(x_test))
            return (x_train, y_train), (x_test, y_test)

    def create_cache_file(self, x, y):
        print('Creating cache file ...')
        x = downsampling(x, self.down_sample_factor, 'axial')
        y = downsampling(y, self.down_sample_factor, 'axial')
        self.indices = np.arange(len(x))
        x_train = x[self.get_ind('train')]
        y_train = y[self.get_ind('train')]
        x_test = x[self.get_ind('test')]
        y_test = y[self.get_ind('test')]
        self.info('w')
        self.save_npy_file(x_train, self.cache_path, 'x_train')
        self.save_npy_file(y_train, self.cache_path, 'y_train')
        self.save_npy_file(x_test, self.cache_path, 'x_test')
        self.save_npy_file(y_test, self.cache_path, 'y_test')
        if self.validation_split > 0:
            x_val = x[self.get_ind('val')]
            y_val = y[self.get_ind('val')]
            self.save_npy_file(x_val, self.cache_path, 'x_val')
            self.save_npy_file(y_val, self.cache_path, 'y_val')
            return (x_train, y_train), (x_test, y_test), (x_val, y_val)
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
            'data_shape':self.shape,
            'complex_data':self.complex_data,
            'load_path':self.dataset_path,
            'cache_path':self.cache_path,
            'seed':self.seed,
            'shuffle_method':self.shuffle_method
            }
        if op is None:
            return saved_var
        elif op == 'r':
            return read_info(file_name, self.cache_path)
        elif op == 'w':
            save_info(saved_var, file_name, self.cache_path)
        else:
            raise ValueError("Expected `r`, `w`, and NoneType for reading, writing and getting information.")