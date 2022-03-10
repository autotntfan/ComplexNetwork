# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 23:33:49 2022

@author: benzener
"""
import scipy.io
import numpy as np
import os
import h5py

def get_variable(path, dtype):
    data = scipy.io.loadmat(path)
    if not isinstance(dtype, str):
        raise AttributeError('key word should be a string')
    if dtype not in data.keys():
        raise KeyError('key word not in data')
    return data.get(dtype)

def reshape_complex_value(x):
    real = np.expand_dims(x.real,axis=-1)
    imag = np.expand_dims(x.imag,axis=-1)
    return np.concatenate((real,imag),axis=-1)


    
def save_variables(file_num=4):
    dir_path = 'D:\simulation_straight'
    file_name = os.listdir(dir_path)
    file_name = file_name[1:] # remove count_file.py
    file_name = sorted(file_name,key=lambda x:int(x.replace("Data_","")[:-12])) # sort file name in order
    data_size = (len(file_name)//file_num,513,513)
    with h5py.File('dataset.hdf5','w') as f:
        for level in range(file_num):
            names = file_name[level::file_num]
            g = f.create_group("level"+str(level+1))
            psf_bb = np.zeros(data_size,dtype=np.complex)
            psf_rf = np.zeros(data_size)
            speckle_bb = np.zeros(data_size,dtype=np.complex)
            speckle_rf = np.zeros(data_size)
            for i,name in enumerate(names):
                file_path = os.path.join(dir_path,name)
                psf_bb[i,:,:] = get_variable(file_path, 'psf_bb')
                psf_rf[i,:,:] = get_variable(file_path, 'psf_rf')
                speckle_bb[i,:,:] = get_variable(file_path, 'speckle_bb')
                speckle_rf[i,:,:] = get_variable(file_path, 'speckle_rf')
                if i%10 == 0:
                    print(i)
            psf_bb = reshape_complex_value(psf_bb)
            speckle_bb = reshape_complex_value(speckle_bb)
            g.create_dataset('psf_bb',psf_bb.shape,dtype='float64',data=psf_bb)
            g.create_dataset('psf_rf',psf_rf.shape,dtype='float64',data=psf_rf)
            g.create_dataset('speckle_rf',speckle_rf.shape,dtype='float64',data=speckle_rf)
            g.create_dataset('speckle_bb',speckle_bb.shape,dtype='float64',data=speckle_bb)
'''
structure:
    --group:level 1
        dataset : psf_bb
                  psf_rf
                  speckle_rf
                  speckle_bb
    --group:level 2
    
'''
        
if __name__ == '__main__':
    save_variables()

    
        
        
    