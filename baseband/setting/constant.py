# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 18:31:41 2022

@author: benzener
"""
import os

MODELPATH = r'./modelinfo' # path of saved model
DATAPATH  = r'./MatlabCheck/simulation_data_2000' # path of simulation data
CACHEPATH = r'./parameters' # path of splitted data


DATASIZE    = (2000,257,257) # simulation data size
k           = 4
FOV         = (16,32)        # field of view, unit in lambda, i.e. (H,W) = (16lambda,32lambda)
EPS         = 1e-16
SOUNDV      = 1540
NELEMENT    = 128
BEAMSPACING = 4
FIRORDER    = 48
COLORSHIT   = ['red','green','blue','purple','orange','cyan','brown','black']
if not os.path.exists(MODELPATH):
    os.mkdir(MODELPATH)
if not os.path.exists(CACHEPATH):
    os.mkdir(CACHEPATH)
    
    
# import os
# import numpy as np

# MODELPATH = r'./modelinfo_diffc' # path of saved model
# DATAPATH  = r'./simulation_data_diffc_4000' # path of simulation data
# CACHEPATH = r'./parameters_diffc' # path of splitted data


# DATASIZE    = (4000,257,257) # simulation data size
# k           = 8
# SPEEDLIST   = np.arange(1460,1620,20)
# FOV         = (16,32)        # field of view, unit in lambda, i.e. (H,W) = (16lambda,32lambda)
# EPS         = 1e-16
# SOUNDV      = 1540
# NELEMENT    = 128
# BEAMSPACING = 4
# FIRORDER    = 48
# COLORSHIT   = ['red','green','blue','purple','orange','cyan','brown','black']
# if not os.path.exists(MODELPATH):
#     os.mkdir(MODELPATH)
# if not os.path.exists(CACHEPATH):
#     os.mkdir(CACHEPATH)
# if len(SPEEDLIST) != k:
#     raise ValueError(f'There are {len(SPEEDLIST)} kinds of SOS but k is {k}.')
