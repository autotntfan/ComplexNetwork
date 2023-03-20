# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:46:45 2023

@author: benzener
"""

from scipy import io
import os
from baseband.setting import constant
from baseband.utils.analysis import mse, mae, ssim, IOU
from baseband.utils.fig_utils import envelope_fig, levelnIOU_fig
import numpy as np
from scipy.signal import correlate
import pandas as pd
import matplotlib.pyplot as plt
from baseband.utils.data_utils import normalization, convert_to_real, focusing, envelope_detection

def load_data(type_):
    soundvs = np.linspace(1460,1600,8)
    datas = np.zeros((8,128,256,2)) 
    for ii, soundv in enumerate(soundvs):
        file_name = 'Data_3_c_' + str(int(soundv)) + '_delay_1.mat' 
        file_path = os.path.join(r'./simulation_data_30test', file_name)
        data = io.loadmat(file_path)
        if type_ == 'psf':
            psf_bb = data.get('psf_bb')
        else:
            psf_bb = data.get('speckle_bb')
        psf_bb = psf_bb[1::2,1:]
        datas[ii] = normalization(convert_to_real(psf_bb))
    return datas.astype(np.float32)


def find_best_sos(model):
    soundvs = np.linspace(1460,1600,8)
    datas = load_data('speckle')
    # ref_path = os.path.join('simulation_data', 'Data_7_delay_1.mat')
    ref_path = os.path.join(r'./simulation_data_30test', 'Data_3_c_1540_delay_1.mat')
    ref = io.loadmat(ref_path)
    ref = ref.get('psf_bb')
    ref = ref[1::2,1:]
    ref_bb = normalization(convert_to_real(ref))
    ref_bb = ref_bb.astype(np.float32)
    envelope_fig(ref_bb)
    sospred = model.predict(datas)

    MSE = np.zeros((8,))
    MAE = np.zeros((8,))
    SSIM = np.zeros((8,))
    for ii in range(8):
        MSE[ii] = mse(sospred[ii],ref_bb, focus=True)
        MAE[ii] = mae(sospred[ii],ref_bb, focus=True)
        SSIM[ii] = ssim(sospred[ii],ref_bb, focus=True)
        envelope_fig(sospred[ii],title_name=f"{soundvs[ii]}m/s")
    table = {
        'MSE':MSE,
        'MAE':MAE,
        'SSIM':SSIM
        }
    df = pd.DataFrame(table, columns = ['MSE','MAE','SSIM'], index=soundvs)
    print('================\n')
    print(df) 
    print(soundvs[np.argmin(MSE)], soundvs[np.argmin(MAE)],soundvs[np.argmax(SSIM)])
    refs = np.repeat(np.expand_dims(ref_bb,axis=0),8,axis=0)
    
    ious, DRs, _, _ = IOU(focusing(sospred), focusing(refs))
    DRtable = {}
    for iDR,DR in enumerate(DRs):
        DRtable[str(DR)] = ious[iDR]
        
    df = pd.DataFrame(DRtable)
    df.columns = DRs
    df.index = soundvs
    df["total"] = np.sum(ious,axis=0)
    print('================\n')
    print("IOU")
    print(df)
    levelnIOU_fig(sospred,refs,np.ones((8,)),np.ones((8,)))
    print(soundvs[np.argmax(np.sum(ious,axis=0))])

def testmetrics():
    soundv = 1540
    soundvs = np.linspace(1460,1600,8)
    ref_path = os.path.join(r'./simulation_data_30test', f'Data_5_c_{soundv}_delay_1.mat')
    ref = io.loadmat(ref_path)
    ref = ref.get('psf_bb')
    ref = ref[1::2,1:]
    ref_bb = normalization(convert_to_real(ref))
    ref_bb = ref_bb.astype(np.float32)
    datas = load_data('psf')

    MSE = np.zeros((8,))
    MAE = np.zeros((8,))
    SSIM = np.zeros((8,))
    ac = []
    for ii in range(8):
        MSE[ii] = mse(datas[ii],ref_bb)
        MAE[ii] = mae(datas[ii],ref_bb)
        SSIM[ii] = ssim(datas[ii],ref_bb)
        ac.append(acorr(np.squeeze(envelope_detection(datas[ii]))))
    plt.figure()
    plt.scatter(soundvs,ac)
    plt.show()
    table = {
        'MSE':MSE,
        'MAE':MAE,
        'SSIM':SSIM
        }
    df = pd.DataFrame(table, columns = ['MSE','MAE','SSIM'], index=soundvs)
    print('================\n')
    print(f"Ref soundv - {soundv}")
    print(df)
    print('Best selection', soundvs[np.argmin(MSE)], soundvs[np.argmin(MAE)],soundvs[np.argmax(SSIM)])
    
    refs = np.repeat(np.expand_dims(ref_bb,axis=0),8,axis=0)
    levelnIOU_fig(datas,refs,np.ones((8,)),np.ones((8,)))
    
def acorr(x):
    total_acorr = np.zeros((x.shape[0],))
    for depth in range(x.shape[0]):
        signal = x[depth,:]
        total_acorr[depth] = np.correlate(signal,signal)
    return np.sum(total_acorr)
       
def testmetrics1():
    soundv = 1540
    soundvs = np.linspace(1460,1600,8)
    datas = load_data('psf')
    MSE = np.zeros((8,1))
    MAE = np.zeros((8,1))
    SSIM = np.zeros((8,1))
    ref_path = os.path.join(r'./simulation_data_30test', f'Data_2_c_{soundv}_delay_1.mat')
    ref = io.loadmat(ref_path)
    ref = ref.get('psf_bb')
    ref = ref[1::2,1:]
    ref_bb = normalization(convert_to_real(ref))
    ref_bb = ref_bb.astype(np.float32)

    MSE = np.zeros((8,))
    MAE = np.zeros((8,))
    SSIM = np.zeros((8,))
    for ii in range(8):
        MSE[ii] = mse(datas[ii],ref_bb)
        MAE[ii] = mae(datas[ii],ref_bb)
        SSIM[ii] = ssim(datas[ii],ref_bb)
    plt.figure()
    plt.plot(soundvs,MSE,c='r')
    plt.title('MSE')
    plt.show()
    plt.figure()
    plt.plot(soundvs,MAE,c='g')
    plt.title('MAE')
    plt.show()
    plt.figure()
    plt.plot(soundvs,SSIM,c='b')
    plt.title('SSIM')
    plt.show()
    table = {
        'MSE':MSE,
        'MAE':MAE,
        'SSIM':SSIM
        }
    df = pd.DataFrame(table, columns = ['MSE','MAE','SSIM'], index=soundvs)
    envelope_fig(ref_bb)
    envelope_fig(datas[1],title_name=f"soundv - {soundv}")
    envelope_fig(datas[4],title_name=f"soundv - {soundvs[4]}")
    print('================\n')
    print(f"Ref soundv - {soundv}")
    print(df)
    print('Best selection', soundvs[np.argmin(MSE)], soundvs[np.argmin(MAE)],soundvs[np.argmax(SSIM)])

def testmetrics2():
    soundvs = np.linspace(1460,1600,8)
    datas = np.zeros((8*30,128,256,2))
    refs = np.zeros((8*30,128,256,2))
    for jj in range(1,31):
        for ii, soundv in enumerate(soundvs):
            file_name = 'Data_' + str(jj) + '_c_' + str(int(soundv)) + '_delay_1.mat' 
            file_path = os.path.join(r'./simulation_data_30test', file_name)
            data = io.loadmat(file_path)
            psf_bb = data.get('psf_bb')
            psf_bb = psf_bb[1::2,1:]
            datas[8*(jj-1) + ii] = normalization(convert_to_real(psf_bb))

    
    for jj in range(30):
        MSE = np.zeros((8,))
        MAE = np.zeros((8,))
        SSIM = np.zeros((8,))
        acorrs = np.zeros((8,))
        for ii in range(8):
            MSE[ii] = mse(datas[8*jj+ii],datas[4+8*jj])
            MAE[ii] = mae(datas[8*jj+ii],datas[4+8*jj])
            SSIM[ii] = ssim(datas[8*jj+ii],datas[4+8*jj])
            acorrs[ii] = acorr(np.squeeze(envelope_detection(datas[8*jj+ii])))

        table = {
            'MSE':MSE,
            'MAE':MAE,
            'SSIM':SSIM
            }
        df = pd.DataFrame(table, columns = ['MSE','MAE','SSIM'], index=soundvs)
        
        refs = np.repeat(np.expand_dims(datas[4+8*jj],axis=0),8,axis=0)
        levelnIOU_fig(datas[8*jj:8*jj+8],refs,np.ones((8,)),np.ones((8,)))
        plt.figure()
        plt.scatter(soundvs, acorrs)
        plt.xlabel('Sound speed (m/s)')
        plt.ylabel('Autocorrelation')
        plt.title(f"Sample - {jj}")
        plt.show()
    # plt.figure(1)
    # plt.scatter(np.repeat(soundvs,30),MSE,c='r')
    # plt.title('MSE')
    # plt.figure(2)
    # plt.scatter(np.repeat(soundvs,30),MAE,c='g')
    # plt.title('MAE')
    # plt.figure(3)
    # plt.scatter(np.repeat(soundvs,30),SSIM,c='b')
    # plt.title('SSIM')
    # plt.show()
    
        print('================\n')
        print(f"Ref soundv - {soundv}")
        print(df)
        print('Best selection', soundvs[np.argmin(MSE)], soundvs[np.argmin(MAE)],soundvs[np.argmax(SSIM)])

def testmetrics3(model, model_name):
    soundvs = np.linspace(1460,1600,8)
    datas = np.zeros((8*30,128,256,2))
    refs = np.zeros((8*30,128,256,2))
    for jj in range(1,31):
        for ii, soundv in enumerate(soundvs):
            file_name = 'Data_' + str(jj) + '_c_' + str(int(soundv)) + '_delay_1.mat' 
            file_path = os.path.join(r'./simulation_data_30test', file_name)
            data = io.loadmat(file_path)
            psf_bb = data.get('speckle_bb')
            psf_bb = psf_bb[1::2,1:]
            datas[8*(jj-1) + ii] = normalization(convert_to_real(psf_bb))
            if ii == 4:
                psf_bb = data.get('psf_bb')
                psf_bb = psf_bb[1::2,1:]
                psf_bb = normalization(convert_to_real(psf_bb))
                refs[8*(jj-1):8*jj] = np.repeat(np.expand_dims(psf_bb,axis=0),8,0)
    datas = datas.astype(np.float32)
    refs = focusing(refs.astype(np.float32))
    sospred = focusing(model.predict(datas))
    bestMSE = np.zeros((31,))
    bestMAE = np.zeros((31,))
    bestSSIM = np.zeros((31,))
    bestIOU = np.zeros((31,))
    for jj in range(30):
        MSE = np.zeros((8,))
        MAE = np.zeros((8,))
        SSIM = np.zeros((8,))
        for ii in range(8):
            envelope_fig(sospred[8*jj+ii], 
                         title_name=f'Prediction - set {jj} - sos {soundvs[ii]}',
                         model_name=model_name,
                         saved_name=f'set_{jj}_{soundvs[ii]}',
                         saved_dir='sos_prediction',
                         show=False)
            MSE[ii] = mse(sospred[8*jj+ii],refs[8*jj+ii])
            MAE[ii] = mae(sospred[8*jj+ii],refs[8*jj+ii])
            SSIM[ii] = ssim(sospred[8*jj+ii],refs[ii+8*jj])
            
        levelnIOU_fig(sospred[8*jj:8*jj+8],refs[8*jj:8*jj+8],np.ones((8,)),np.ones((8,)))
        table = {
            'MSE':MSE,
            'MAE':MAE,
            'SSIM':SSIM
            }
        df = pd.DataFrame(table, columns = ['MSE','MAE','SSIM'], index=soundvs)
        print('================\n')
        print(f"Ref soundv - 1540m/s - {jj}")
        print(df)
        print('Best selection', soundvs[np.argmin(MSE)], soundvs[np.argmin(MAE)],soundvs[np.argmax(SSIM)])
        ious, DRs, _, _ = IOU(sospred[8*jj:8*jj+8],refs[8*jj:8*jj+8])
        DRtable = {}
        
        for iDR,DR in enumerate(DRs):
            DRtable[str(DR)] = ious[iDR]       
        df = pd.DataFrame(DRtable)
        df.columns = DRs
        df.index = soundvs
        df["total"] = np.sum(ious,axis=0)
        print('================\n')
        print(f"IOU - {jj}")
        print(df)
        print(soundvs[np.argmax(np.sum(ious,axis=0))])
        bestMSE[jj] = soundvs[np.argmin(MSE)]
        bestMAE[jj] = soundvs[np.argmin(MAE)]
        bestSSIM[jj] = soundvs[np.argmax(SSIM)]
        bestIOU[jj] = soundvs[np.argmax(np.sum(ious,axis=0))]
    bestMSE[-1] = np.sum(bestMSE==1540)
    bestMAE[-1] = np.sum(bestMAE==1540)
    bestSSIM[-1] = np.sum(bestSSIM==1540)
    bestIOU[-1] = np.sum(bestIOU==1540)

    table = {
        'MSE':bestMSE,
        'MAE':bestMAE,
        'SSIM':bestSSIM,
        'IOU':bestIOU
        }
    df = pd.DataFrame(table, columns = ['MSE','MAE','SSIM','IOU'])
    print('================\n')
    print(df)
# if __name__ == '__main__':
    # testmetrics1()
