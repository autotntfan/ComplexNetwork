# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 21:13:56 2022

@author: benzener
"""
import complexnn
import os
import time
import numpy             as np
import tensorflow        as tf
import seaborn           as sns
import matplotlib.pyplot as plt
import scipy.signal      as Signal
from scipy    import io
from datetime import datetime
from computation import BasedCompute, Difference
from DisplayedImg import Fig

DIR_SAVED          = r'./modelinfo'
# DIR_SIMULATION     = r'./simulation_straight'
DIR_SIMULATION     = r'./simulation_data'
DATA_SIZE = (2000,257,257)
B = BasedCompute()
D = Difference()
F = Fig()

def get_custom_object():
    custom_object = {
        'ComplexConv2D':complexnn.conv_test.ComplexConv2D,
        'ComplexBatchNormalization':complexnn.bn_test.ComplexBatchNormalization,
        'ComplexMSE':complexnn.loss.ComplexMSE,
        'ctanh':complexnn.activation.ctanh,
        'FLeakyReLU': complexnn.activation.FLeakyReLU
        }
    return custom_object

def read_info(model_name):
    '''
     reading the information saved by save_info function
     this function is in order to check whether the given
     augments are compatible with loaded model
    '''
    saved_dir = os.path.join(DIR_SAVED, model_name)
    file_name = model_name + '_parameters.txt'
    if not os.path.exists(saved_dir):
        raise FileNotFoundError('file does not exist')
    saved_path = os.path.join(saved_dir, file_name)
    with open(saved_path, 'r') as f:
        content = f.read() # type of content is string
    return eval(content) # convert string to dict

def save_info(model_name, saved_var):
    '''
    save the information of parameters, included 
    epoch. seed. and so on
    Args:
        model_name: saved path
        saved_var: parameters to be preserved
    '''
    saved_dir = os.path.join(DIR_SAVED, model_name)
    file_name = model_name + '_parameters.txt'
    if not os.path.exists(saved_dir):
        try:
            os.mkdir(saved_dir)
        except FileNotFoundError:
            os.makedirs(saved_dir)
    saved_path = os.path.join(saved_dir, file_name)
    with open(saved_path, 'w') as f:
        f.write(str(saved_var))
    
def save_model(model, history, name):
    saved_dir = os.path.join(DIR_SAVED, name)
    model_name = name + '.h5'      # model
    model_figname = name + 'arc.png'  # model architecture 
    if not os.path.exists(saved_dir):
        try:
            os.mkdir(saved_dir)
        except FileNotFoundError:
            os.makedirs(saved_dir)
    model_saved_path = os.path.join(saved_dir, model_name)
    model_figpath = os.path.join(saved_dir, model_figname)
    # plot and save model architecture
    tf.keras.utils.plot_model(model, to_file=model_figpath, show_shapes=True, show_layer_names=True, dpi=900)
    model.save(model_saved_path)
    history_name = os.path.join(saved_dir, name + 'history.txt') # loss value per epoch
    with open(history_name, 'w') as f:
        f.write(str(history))
    plt.figure()
    plt.plot(history['loss'], label='training loss')
    try:
        plt.plot(history['val_loss'], label='val loss')    
    except KeyError:
        print('No validation')
    finally:
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig(os.path.join(saved_dir,name+'.png'))
        plt.show()
        
def get_default(complex_network=True):
    batch = 8
    if complex_network:
        decimation = 2
        size = (3,6)
        loss = 'ComplexMSE'
    else:
        decimation = 1
        size = (6,6)
        loss = 'MSE'
    return decimation, size, loss, batch

        
    
def inference(model, testset):
    time_collection = []
    for i in range(10):
        _ = model.predict(testset[i:i+10])
    for i in range(10):
        s = time.time()
        _ = model.predict(testset)
        e = time.time()
        time_collection.append(e-s)
    return np.mean(time_collection)/testset.shape[0]

    
def multilevel_projection(pred, truth, OBJ, model_name=None):
    pred = B.focusing(pred)
    truth = B.focusing(truth)
    proj_pred = B.projection(pred, 0)
    proj_true = B.projection(truth, 0)
    levels = np.zeros(pred.shape[0], int)
    color = ['red','green','blue','black']
    labels = ['level1','level2','level3','level4']
    for i in range(pred.shape[0]):
        levels[i], _ = OBJ.find_level(i, train=False)
    for i in range(1,5):
        plt.figure()
        pred_proj = np.mean(proj_pred[levels==i],axis=0)
        true_proj = np.mean(proj_true[levels==i],axis=0)
        plt.plot(pred_proj, color=color[i-1],label=labels[i-1]+'p')
        plt.plot(true_proj, color=color[i-1], linestyle='dashed',label=labels[i-1]+'t')
        plt.title('level' + str(i))
        plt.legend()
        plt.show()
    plt.figure
    for i in range(1,5):
        pred_proj = np.mean(proj_pred[levels==i],axis=0)
        plt.plot(pred_proj, color=color[i-1],label=labels[i-1])
    plt.title('prediction')
    plt.legend()
    plt.show()
    plt.figure()
    for i in range(1,5):
        true_proj = np.mean(proj_true[levels==i],axis=0)
        plt.plot(true_proj, color=color[i-1], linestyle='dashed',label=labels[i-1])
    plt.title('truth')
    plt.legend()
    plt.show()

def multilevel_projection_m(pred, truth, OBJ):
    pred = B.focusing(pred)
    truth = B.focusing(truth)
    proj_pred = B.projection(pred, 0)
    proj_true = B.projection(truth, 0)
    levels = np.zeros(pred.shape[0], int)
    pred_proj = np.zeros((4,pred.shape[2]))
    true_proj = np.zeros((4,pred.shape[2]))
    for i in range(pred.shape[0]):
        levels[i], _ = OBJ.find_level(i, train=False)
    for i in range(1,5):
        pred_proj[i-1,:] = np.mean(proj_pred[levels==i],axis=0)
        true_proj[i-1,:] = np.mean(proj_true[levels==i],axis=0)
    return pred_proj,true_proj

def checkLBP(pred, truth, OBJ, model_name=None):
    pred = B.focusing(pred)
    truth = B.focusing(truth)
    proj_pred = B.projection(pred, 0)
    proj_true = B.projection(truth, 0)
    levels = np.zeros(pred.shape[0], int)
    inds = np.zeros(pred.shape[0], int)
    delay = np.zeros((pred.shape[0],128))
    for i in range(pred.shape[0]):
        levels[i], inds[i] = OBJ.find_level(i, train=False)
        delay[i] = B.get_delaycurve(inds[i])
    LBPD = D.BPD(pred, truth)
    delay = delay[levels==4]
    proj_pred = proj_pred[levels==4]
    proj_true = proj_true[levels==4]
    LBPD = LBPD[levels==4]
    inds = inds[levels==4]
    ii = np.argsort(LBPD)
    delay = delay[ii]
    proj_pred = proj_pred[ii]
    proj_true = proj_true[ii]
    pred = pred[ii]
    truth = truth[ii]
    LBPD = LBPD[ii]
    inds = inds[ii]
    for i in range(np.size(LBPD)):
        plt.figure()
        plt.plot(proj_pred[i], label='pred')
        plt.plot(proj_true[i], linestyle='dashed',label='true')
        plt.title('projection' + str(i) + '_' + str(inds[i]) + '_' + str(LBPD[i]))
        plt.legend()
        B.save_fig(model_name, 'L4_' + str(i) + '_' + str(inds[i]) + 'proj', 'L4projection')
        plt.show()
        F.envelope_fig(pred[i], title_name=str(inds[i]) + '_' + str(i),model_name=model_name, saved_name='L4p'+ str(i) + '_' + str(inds[i]),saved_dir='L4projection')
        F.envelope_fig(truth[i], title_name=str(inds[i])+ '_' + str(i),model_name=model_name, saved_name='L4t'+ str(i) + '_' + str(inds[i]),saved_dir='L4projection')
        F.delay_fig(delay[i], title_name=str(inds[i])+ '_' + str(i),model_name=model_name, saved_name='L4_' + str(i) + '_' + str(inds[i]) + 'delay', saved_dir='L4projection')
    plt.figure()
    plt.plot(np.mean(proj_pred[:i//4],axis=0), label='0.25p', color='green')
    plt.plot(np.mean(proj_pred[i//4:2*i//4], axis=0), label='0.50p', color='blue')
    plt.plot(np.mean(proj_pred[2*i//4:3*i//4],axis=0), label='0.75p', color='red')
    plt.plot(np.mean(proj_pred[3*i//4:],axis=0), label='1.00p', color='black')
    plt.plot(np.mean(proj_true[:i//4], axis=0), linestyle='dashed', label='0.25t', color='green')
    plt.plot(np.mean(proj_true[i//4:2*i//4], axis=0), linestyle='dashed', label='0.50t', color='blue')
    plt.plot(np.mean(proj_true[2*i//4:3*i//4], axis=0), linestyle='dashed', label='0.75t', color='red')
    plt.plot(np.mean(proj_true[3*i//4:], axis=0), linestyle='dashed', label='1.00t', color='black')
    plt.legend()
    plt.show()
    
def multilevel_IOU(signal1, signal2, OBJ, training=False, model_name=None):
    assert signal1.shape == signal2.shape
    signal1 = B.focusing(signal1)
    signal2 = B.focusing(signal2)
    levels = np.zeros(signal1.shape[0], int)
    inds = np.zeros(signal1.shape[0], int)
    iou, DRs, mask1, mask2 = Difference().IOU(signal1,signal2)
    color = ['red','green','blue','black']
    for i in range(iou.shape[1]):
        levels[i], inds[i] = OBJ.find_level(i, train=training)
        envelope_pred = B.reduce_dim(B.envelope_detection(signal1[i], np.max(DRs)))
        envelope_ref = B.reduce_dim(B.envelope_detection(signal2[i], np.max(DRs)))
        plt.figure(figsize=(20,20))
        plt.subplot(5,2,1)
        plt.imshow(envelope_pred, cmap='gray', vmin=0, vmax=np.max(DRs),aspect='auto')
        plt.title('prediction')
        plt.subplot(5,2,2)
        plt.imshow(envelope_ref, cmap='gray', vmin=0, vmax=np.max(DRs),aspect='auto')
        plt.title('true')
        plt.subplot(5,2,3)
        plt.imshow(mask1[0,i], cmap='gray', vmin=0, vmax=1, aspect='auto')
        plt.title('0dB')
        plt.subplot(5,2,4)
        plt.imshow(mask2[0,i], cmap='gray', vmin=0, vmax=1, aspect='auto')
        plt.title('0dB_' + str(iou[0,i]))
        plt.subplot(5,2,5)
        plt.imshow(mask1[1,i], cmap='gray', vmin=0, vmax=1, aspect='auto')
        plt.title('20dB')
        plt.subplot(5,2,6)
        plt.imshow(mask2[1,i], cmap='gray', vmin=0, vmax=1, aspect='auto')
        plt.title('20dB_' + str(iou[1,i]))
        plt.subplot(5,2,7)
        plt.imshow(mask1[2,i], cmap='gray', vmin=0, vmax=1, aspect='auto')
        plt.title('40dB')
        plt.subplot(5,2,8)
        plt.imshow(mask2[2,i], cmap='gray', vmin=0, vmax=1, aspect='auto')
        plt.title('40dB_' + str(iou[2,i]))
        plt.subplot(5,2,9)
        plt.imshow(mask1[3,i], cmap='gray', vmin=0, vmax=1, aspect='auto')
        plt.title('60dB')
        plt.subplot(5,2,10)
        plt.imshow(mask2[3,i], cmap='gray', vmin=0, vmax=1, aspect='auto')
        plt.title('60dB_' + str(iou[3,i]))
        B.save_fig(model_name, 'IOU_L' + str(levels[i]) + '_' + str(inds[i]), 'IOU')
        plt.show()
    start = 0
    count = np.zeros((iou.shape[0],1))
    title_names = ['I <= -60dB', '-60dB < I <= -40dB', '-40dB < I <= -20dB', '-20dB < I <= 0dB']
    for level in range(1,5):
        level_n_iou = iou[:,levels==level]
        end = start + level_n_iou.shape[1]
        for ii in range(iou.shape[0]):
            plt.figure(ii+1)
            plt.scatter(np.arange(start,end),level_n_iou[ii,:],c=color[level-1])
            plt.ylim((0.0,1.0))
            plt.xlabel('Sample')
            plt.ylabel('IOU')
            plt.title(title_names[ii])
            B.save_fig(model_name, 'IOU_' + str(ii))
            count[ii,:] = count[ii,:] + sum(level_n_iou[ii,:]>0.5);
        start = end

    