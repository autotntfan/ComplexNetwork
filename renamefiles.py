# -*- coding: utf-8 -*-
"""
Created on Wed May  4 12:01:58 2022

@author: benzener
"""
import os

def rename_files(old_dir=r'simulation_data2', new_dir=r'simulation_data'):
    oldfile_list = os.listdir(old_dir)
    num_newfiles = file_counts(oldfile_list, type_='mat')/4
    for i, name in enumerate(oldfile_list):
        Data, number, delay, K = name.split('.')[0].split('_')
        if name.split('.')[-1] in {'png'}:
            new_name = Data + '_' + str(int(num_newfiles) + int(number)) + '_' + delay + '_' + K + '.png'
        elif name.split('.')[-1] in {'mat'}:
            new_name = Data + '_' + str(int(num_newfiles) + int(number)) + '_' + delay + '_' + K + '.mat'
        else:
            continue
        old_path = os.path.join(old_dir, name)
        new_path = os.path.join(new_dir, new_name)
        os.rename(old_path, new_path)
        if i%100 == 0:
            print(i+1)
    
def file_counts(file_list, type_='mat'):
    count = 0
    for name in file_list:
        if name.split('.')[-1] == type_:
            count = count + 1
    return count
        
            