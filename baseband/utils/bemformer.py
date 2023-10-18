"""
Created on Tue May  9 13:47:45 2023

@author: benzener
"""
from scipy import io, signal
import numpy as np
import cv2
from baseband.utils.fig_utils import envelope_fig

def get_data_info(file):
    data = io.loadmat(file)
    channeldata = np.squeeze(data.get('channeldata'))
    soundv = np.squeeze(data.get('soundv')).item()
    pitch = np.squeeze(data.get('pitch')).item()
    f0 = np.squeeze(data.get('f0')).item()
    fs = np.squeeze(data.get('fs')).item()
    Noffset = np.squeeze(data.get('Noffset')).item()
    ptloc = np.squeeze(data.get('ptloc')).item()
    return channeldata, soundv, pitch, f0, fs, Noffset, ptloc
    
def beamform(channeldata,
             scat_space,
             soundv,
             beamformv,
             pitch,
             f0,
             fs,
             Noffset,
             ptloc,
             Npx,
             Npz,
             f_num=2,
             beamspacing=4,
             gain=0,
             DR=60,
             psfsegx=32/2,
             psfsegz=16/2):    
    
    channeldataT = np.transpose(channeldata,(2,0,1)) # permutate [z, rx, tx] to [tx, z, rx]
    # Note: This permutation is in order to help to do beamform last two dimesion [H,W]
    Nelements, Nsample, _ = channeldataT.shape
    fulldataset  = np.zeros((Nelements, Nsample+1, Nelements))
    fulldataset[:, :-1,:] = channeldataT
    Lambda = beamformv/f0
    dz0 = beamformv/fs/2
    FOVx = int(np.ceil(psfsegx*2*Lambda/pitch).item())
    standardz= soundv/fs/2*(Noffset + np.linspace(0,Nsample-1,Nsample))
    x_elem   = np.linspace(-(Nelements-1)/2,(Nelements-1)/2, Nelements)*pitch
    x_range  = ((FOVx-1)/2 + 1/beamspacing*(beamspacing-1)/2)*pitch
    x_aline  = np.linspace(-x_range,x_range,beamspacing*FOVx)
    
    xx_elem  = np.repeat(x_elem, Nsample*FOVx*beamspacing)
    xx_elem  = np.reshape(xx_elem, [Nelements, Nsample, FOVx*beamspacing])
    xx_aline = np.tile(x_aline, Nsample).reshape(1,Nsample, -1)
    xx_aline = np.repeat(xx_aline, Nelements, axis=0)
    z        = dz0*(Noffset + np.linspace(0,Nsample-1,Nsample))
    
    channel_index1 = np.repeat(np.arange(0, Nelements), (Nsample+1)*Nelements).reshape(Nelements, Nsample+1, Nelements)
    channel_index3 = np.tile(np.arange(0, Nelements), (Nsample+1)*Nelements).reshape(Nelements, Nsample+1, Nelements)
    
    
    # distance = np.sqrt((xx_aline - xx_elem)**2 + np.reshape(np.repeat(z,FOVx*beamspacing*Nelements), [-1, FOVx*beamspacing, Nelements])**2)
    distance = np.abs(xx_aline - xx_elem + 1j*np.reshape(np.repeat(z,FOVx*beamspacing), [1, Nsample, FOVx*beamspacing]))
    z_pt_start_ind = np.argmin(np.abs(standardz - (ptloc - psfsegz*Lambda) ))
    z_pt_end_ind = np.argmin(np.abs(standardz - (ptloc + psfsegz*Lambda)))
    beamformed_region_z = z[z_pt_start_ind:z_pt_end_ind+1];
    channel_index1 = channel_index1[:,z_pt_start_ind:z_pt_end_ind+1,:]
    channel_index3 = channel_index3[:,z_pt_start_ind:z_pt_end_ind+1,:]
    distance = distance[:,z_pt_start_ind:z_pt_end_ind+1,:]

    psf_rf = np.zeros((len(beamformed_region_z), len(x_aline)))
    for Nline in range(len(x_aline)):
        f_num_mask_rx = np.expand_dims(beamformed_region_z,axis=1)/np.expand_dims((2*np.abs(x_aline[Nline] - x_elem)), axis=0) > f_num
        f_num_mask_tx = np.expand_dims(f_num_mask_rx.T, axis=2)
        f_num_mask_rx = np.expand_dims(f_num_mask_rx, axis=0)
        Nlinedistance = distance[:,:,Nline:Nline+1] + np.expand_dims(distance[:,:,Nline].T, axis=0)
        channel_ind = np.ceil(Nlinedistance/beamformv*fs - Noffset).astype(np.int32)
        channel_ind[channel_ind > Nsample] = Nsample + 1
        channel_ind[channel_ind < 1] = Nsample + 1
        channel_ind = (channel_index1, channel_ind, channel_index3)
        psf_rf[:,Nline] = np.sum(f_num_mask_rx*f_num_mask_tx*fulldataset[channel_ind], axis=(0,2)); 

    newz = np.interp(np.linspace(z_pt_start_ind,z_pt_end_ind,Npz), np.arange(z_pt_start_ind,z_pt_end_ind+1), beamformed_region_z)
    # newx = np.interp(np.arange(x_pt_start_ind,x_pt_end_ind+1), beamformed_region_x_aline, np.linspace(x_pt_start_ind,x_pt_end_ind,Npx))
    # depth = z(z_pt_start_ind);
    # dx = newx[1] - newx[0];
    dz = newz[1] - newz[0];
    newfs = beamformv/2/dz;
    lpf = signal.firwin(48, f0/(newfs/2))
        
    psf_rf = cv2.resize(psf_rf, (Npz,Npx));
    psf_bb = np.zeros_like(psf_rf, dtype=np.complex64)
    for ii in range(psf_rf.shape[1]):
        psf_bb[:,ii] = signal.convolve(psf_rf[:,ii]*np.exp(-1j*2*np.pi*f0*np.arange(0,psf_rf.shape[1])/newfs), lpf, 'same')
    envelope_fig(psf_bb, DR, gain)
    Nsx = (Npx - 1)*2 + 1;
    Nsz = (Npz - 1)*2 + 1;
    speckle_rf = signal.convolve2d(scat_space, psf_rf, 'same')    
    speckle_rf = speckle_rf[int((Nsz+1)/2 - (Npz-1)/2):int((Nsz+1)/2 + (Npz-1)/2), int((Nsx+1)/2 - (Npx-1)/2):int((Nsx+1)/2 + (Npx-1)/2)]
    speckle_rf = cv2.resize(speckle_rf, (Npz,Npx))
    speckle_bb = np.zeros_like(speckle_rf, dtype=np.complex64)
    for ii in range(speckle_rf.shape[1]):
        speckle_bb[:,ii] = signal.convolve(speckle_rf[:,ii]*np.exp(-1j*2*np.pi*f0*np.arange(0,speckle_rf.shape[1])/newfs), lpf, 'same')
    return psf_bb, speckle_bb

def generate_scat_space(Npz,Npx):
    Nsx = (Npx - 1)*2 + 1;
    Nsz = (Npz - 1)*2 + 1;
    den         = 0.05 + 0.45*np.random.rand(1)
    Nscat       = int(np.round(den*Nsz*Nsx))
    scat_space  = np.zeros((Nsz,Nsx))
    scat_indice = np.unravel_index(np.random.randint(0, np.size(scat_space)-1,[1,Nscat]), scat_space.shape)
    scat_space[scat_indice] = np.random.randn(Nscat,)
    scat_space = np.reshape(scat_space, [Nsz, Nsx]);
    return scat_space

if __name__  == '__main__':
    channeldata, soundv, pitch, f0, fs, Noffset, ptloc = get_data_info('./MatlabCheck/simulation_data_diffc_channeldata/Data_1540.mat')
    beamformv = 1540
    Npx = 257
    Npz = 257
    f_num=2
    beamspacing=4
    gain=0
    DR=60
    psfsegx=32/2
    psfsegz=16/2 
    scat_space = generate_scat_space(257,257)
    psf_bb, speckle_bb = beamform(channeldata,scat_space,soundv,1540,pitch,f0,fs,Noffset,ptloc,257,257) 