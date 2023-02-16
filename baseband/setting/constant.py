# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 18:31:41 2022

@author: benzener
"""
import os

MODELPATH = r'./modelinfo' # path of saved model
DATAPATH  = r'./simulation_data' # path of simulation data
CACHEPATH = r'./parameters' # path of splitted data

DATASIZE    = (2000,257,257) # simulation data size
FOV         = (16,32)        # field of view, unit in lambda, i.e. (H,W) = (16lambda,32lambda)
EPS         = 1e-16
SOUNDV      = 1540
NELEMENT    = 128
BEAMSPACING = 4
FIRORDER    = 48

if not os.path.exists(MODELPATH):
    os.mkdir(MODELPATH)
if not os.path.exists(CACHEPATH):
    os.mkdir(CACHEPATH)
