# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 18:31:41 2022

@author: benzener
"""
import os

MODELPATH = r'./modelinfo' # path of saved model
DATAPATH  = r'./simulation_data' # path of simulation data
CACHEPATH = r'./parameters' # path of splitted data

DATASIZE  = (2000,257,257) # simulation data size
EPS       = 1e-16
SOUNDV    = 1540
NELEMENT  = 128

if not os.path.exists(MODELPATH):
    os.mkdir(MODELPATH)
if not os.path.exists(CACHEPATH):
    os.mkdir(CACHEPATH)
