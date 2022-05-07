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


DIR_SAVED          = r'./modelinfo'
# DIR_SIMULATION     = r'./simulation_straight'
DIR_SIMULATION     = r'./simulation_data'
DATA_SIZE = (2000,257,257)

def check_data_range(x):
    print(f'the largest value is {np.max(x)} and the smallest one is {np.min(x)}')
    
def show_fig(img, ind=None, title_=None, DR=None, name=None):
    '''
    show_fig function shows the B-mode image in dB scale if DR 
    exists otherwise linear scale. More known arguments, such
    as ind, title_ and name, it gives you more information.
    
    
    Args:
        img: numpy array 
            Displayed image array with shape [1,H,W,C] or [H,W,C]
        ind: int 
            The index or the n-th data, e.g. 'Data_179_delay_2' 
            is the 713-rd data.
        title_: string
            The title of displayed image
        DR: Dynamic range
        name: string
            the name of the desired model, in order to get the
            axis of the image.
    '''
    # check image dimension is [H,W,C] or [1,H,W,C]
    if img.ndim == 4:
        if img.shape[0] != 1:
            raise ValueError('unable to support multiple images')
        img = img[0] # [1,H,W,C] -> [H,W,C]
    elif img.ndim != 3: # only support [1,H,W,C] or [H,W,C]
        raise ValueError('invalid image shape')
    assert img.ndim == 3
    plt.figure(dpi=300)
    if ind is None:
        axis = None
    else:
        # obtain depth and lateral position
        axis = _get_axis(img, ind)
        plt.xlabel('Lateral position (mm)')
        plt.ylabel('Depth (mm)')
    # envelope_detection: [H,W,C] -> [H,W]
    envelope_img = envelope_detection(img, DR)
    # plt.imshow() only supports [H,W] for grayscale.
    assert envelope_img.ndim == 2
    if DR is not None:
        plt.imshow(envelope_img,
                    cmap='gray',
                    vmin=0,vmax=DR,
                    extent=axis,
                    aspect='auto')
        plt.colorbar()
    else:
        plt.imshow(envelope_img, cmap='gray',extent=axis,aspect='auto')
    if title_ is not None:
        assert isinstance(title_, str)
        plt.title(title_)
    if name is not None:
        saved_name = os.path.join(DIR_SAVED, name , title_ + str(ind) + '.png')
        plt.savefig(saved_name, dpi=300)
    plt.show()

def envelope_detection(signal, DR=None):
    '''
    Parameters
    ----------
    signal : float32, numpy array with shape [N,H,W,C] or [H,W,C]
    DR : int, dynamic range. The default is None.
    Returns
    -------
    envelop-detected signal in float32. Its shape is [N,H,W] or
    [H,W] depending on the input shape.
    '''
    # check dtype
    if not np.isreal(signal).all():
        raise ValueError('signal must be an one-channel or two-channel real-valued array')
    # output shape = [H,W] if input shape is [H,W,C],
    # output shape = [N,H,W] if input shape is [N,H,W,C]
    reduce_dim = False 
    # check rank and expand to [N,H,W,C]
    if signal.ndim == 3: # [H,W,C] -> [N,H,W,C]
        signal = np.expand_dims(signal, axis=0)
        reduce_dim = True
    shape = signal.shape # shape = [N,H,W,C]
    assert len(shape) == 4
    if shape[-1] == 2:
        # if complex, envelope = absolute value
        envelope = np.sqrt(signal[:,:,:,0]**2 + signal[:,:,:,1]**2)
    elif shape[-1] == 1:
        # if real, envelope = absolute after hilbert transform
        envelope = np.abs(Signal.hilbert(signal, axis=1)).reshape(shape[:-1])
    else:
        raise ValueError('signal channels must be one or two')
    ratio = np.max(envelope, axis=(1,2),keepdims=True)
    envelope = envelope/ratio
    if DR is None:
        if reduce_dim:
            return envelope[0] # [H,W]
        else:
            return envelope # [N,H,W]
    else:
        dB_img = 20*np.log10(envelope + 1e-16) + DR
        if reduce_dim:
            return dB_img[0] # [H,W]
        else:
            return dB_img # [N,H,W]

def set_env():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

def get_custom_object():
    custom_object = {
        'ComplexConv2D':complexnn.conv_test.ComplexConv2D,
        'ComplexBatchNormalization':complexnn.bn_test.ComplexBatchNormalization,
        'ComplexMSE':complexnn.loss.ComplexMSE,
        'ctanh':complexnn.activation.ctanh,
        'FLeakyRELU_threshold': 0.05,
        'FLeakyReLU': complexnn.activation.FLeakyReLU
        }
    return custom_object

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
    finally:
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig(os.path.join(saved_dir,name+'.png'))
        plt.show()
def save_metrics(envelope_pred, envelope_true, name):
    save_dir = os.path.join(DIR_SAVED, name)
    file_name = os.path.join(save_dir, name + '_metrics.txt')
    mse = tf.losses.MeanSquaredError()
    metrics = {
        'mse':tf.reduce_mean(mse(envelope_pred,envelope_true)).numpy(),
        'ssim':tf.reduce_mean(tf.image.ssim(envelope_pred,
                                            envelope_true,
                                            max_val=1,
                                            filter_size=7)).numpy(),
        'ms_ssim':tf.reduce_mean(tf.image.ssim_multiscale(envelope_pred,
                                                          envelope_true,
                                                          max_val=1,
                                                          filter_size=7)).numpy()
        }
    with open(file_name,'w') as f:
        f.write(str(metrics))
     
def _get_axis(img, ind, fs=False):
    assert img.ndim == 3
    r, c = img.shape[:2]
    if (ind+1)%4 == 0:
        level = 4
    else:
        level = (ind+1)%4
    file_name = 'Data_' + str(ind//4 + 1) + '_delay_' + str(level) + '.mat'
    print(file_name)
    file_path = os.path.join(DIR_SIMULATION, file_name)
    data = io.loadmat(file_path)
    dx = data.get('dx') * (DATA_SIZE[2]/img.shape[1])
    dz = data.get('dz') * (DATA_SIZE[1]/img.shape[0])
    depth = data.get('depth')/2
    x_axis = np.linspace(0,dx*c-dx,c) * 1e3 # [mm]
    z_axis = np.linspace(0,dz*r-dz,r) * 1e3 + depth * 1e3 # [mm]
    xmin, xmax = (np.min(x_axis), np.max(x_axis))
    zmin, zmax = (np.min(z_axis), np.max(z_axis))
    if fs:
        return 1/(2*dz/1540).reshape([-1])
    else:
        return (xmin, xmax, zmax, zmin)
    
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

def projection(signal, ref=None, name=None, DR=60, direction='lateral'):
    '''
    project along axis. 
    lateral projection if axis = H (height).
    axial projection if axis = W (width).
    e.g. input signal with shape (1,128,256,1)
        Expected output shape for lateral projection is (256,),
        which means the np.max along the second axis (axis=1).
        Axial projection is (128,) when np.max along the third
        axis (axis=2).
        
    Parameters
    ----------
    signal : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    if direction not in {'lateral','axial'}:
        raise ValueError("direction only along 'lateral' or 'axial' ")
    if signal.ndim == 4:
        axis = 1 if direction == 'lateral' else 2
    elif signal.ndim < 2 or signal.ndim > 4:
        raise ValueError(f'Unsupport dimension {signal.ndim}')
    else:
        axis = 0 if direction == 'lateral' else 1   
    plt.figure()
    plt.plot(np.max(envelope_detection(signal,DR), axis, initial=0))
    if ref is not None:
        assert ref.shape == signal.shape
        plt.plot(np.max(envelope_detection(ref,DR), axis, initial=0))
        plt.legend(['pred','true'])
    if name is not None:
        saved_name = os.path.join(DIR_SAVED, name ,name + direction + '.png')
        plt.savefig(saved_name, dpi=300)
    plt.show()


def show_fft(signal, ind, Aline=False):
    # signal shape = [H,W,C]
    fs = _get_axis(signal, ind, fs=True)
    # signal shape = [H,W]
    signal = signal.reshape(signal.shape[:-1])
    if Aline:
        center = signal.shape[1]//2
        Signal = np.abs(np.fft.fftshift(np.fft.fft(signal[:,center], axis=0)))
    else:
        Signal = np.abs(np.fft.fftshift(np.fft.fft(signal, axis=0)))
    freq_axis = np.linspace(-fs/2,fs/2,Signal.shape[0])/1e6
    plt.figure()
    plt.plot(freq_axis,Signal)
    plt.xlabel('MHz')
    plt.show()
 
def show_angle(signal, threshold=None, name=None):
    if signal.ndim == 4:
        if signal.shape[0] != 1:
            raise ValueError('Only support one image')
        else:
            signal = signal.reshape(signal.shape[1:])
    assert signal.ndim == 3
    if signal.shape[-1]%2:
        signal = Signal.hilbert(signal, axis=0).reshape(signal.shape[:-1])
    else:
        signal = signal[:,:,0] + 1j*signal[:,:,1]
    angle = np.abs(np.angle(signal))
    sns.heatmap(angle, cmap='hot')
    if name is not None:
        saved_name = os.path.join(DIR_SAVED, name , name + '_angledistribution.png')
        plt.savefig(saved_name, dpi=300)
    plt.show()


def distribution(signal):
    if signal.ndim == 4:
        if signal.shape[0] != 1:
            raise ValueError('Only support one image')
        else:
            signal = signal.reshape(signal.shape[1:])
    assert signal.ndim == 3
    if signal.shape[-1]%2:
        signal = Signal.hilbert(signal, axis=0).reshape(signal.shape[:-1])
    else:
        signal = signal[:,:,0] + 1j*signal[:,:,1]
    signal = signal/np.max(np.abs(signal))
    real = np.real(signal)
    imag = np.imag(signal)
    return real, imag

def show_distribution(signal1, signal2=None):
    real1, imag1 = distribution(signal1)
    if signal2 is not None:
        real2, imag2 = distribution(signal2)
        plt.figure()
        sns.heatmap(np.abs(real1-real2),cmap='Greys')
        plt.title('real part')
        plt.show()
        plt.figure()
        sns.heatmap(np.abs(imag1-imag2),cmap='Greys')
        plt.title('imag part')
        plt.show()
    else:
        plt.figure()
        sns.heatmap(real1, vmin=-1, vmax=1, cmap='Greys')
        plt.title('real part')
        plt.show()
        plt.figure()
        sns.heatmap(imag1, vmin=-1, vmax=1, cmap='Greys')
        plt.title('imag part')
        plt.show()
    
            
            
def phase_diff(y_true, y_pred, threshold=None, name=None):
    assert y_true.shape == y_pred.shape
    if y_true.ndim == 4:
        if y_true.shape[0] != 1:
            raise ValueError('Only support one image')
        else:
            y_true = y_true.reshape(y_true.shape[1:])
            y_pred = y_true.reshape(y_true.shape[1:])
    assert y_true.ndim == 3
    if y_true.shape[-1]%2:
        y_true = Signal.hilbert(y_true, axis=0).reshape(y_pred.shape[:-1])
        y_pred = Signal.hilbert(y_pred, axis=0).reshape(y_pred.shape[:-1])
    else:
        y_true = y_true[:,:,0] + 1j*y_true[:,:,1]
        y_pred = y_pred[:,:,0] + 1j*y_pred[:,:,1]
    angle_err = np.abs(np.angle(y_true) - np.angle(y_pred))
    sns.heatmap(angle_err, cmap='hot')
    if name is not None:
        saved_name = os.path.join(DIR_SAVED, name , name + '_anglediff.png')
        plt.savefig(saved_name, dpi=300)
    plt.show()
    if threshold is not None:
        plt.figure()
        plt.imshow((angle_err<threshold).astype(np.float32),
                   cmap='gray', vmax=1, vmin=0)
        if name is not None:
            saved_name = os.path.join(DIR_SAVED, name , name + '_binarydiff.png')
            plt.savefig(saved_name, dpi=300)
        plt.show()

def multimodel(test_img):
    model_set = {
        'complexmodel_Notforward_300_SSIM_LeakyReLU_22042022', # two-branch SSIM
        'complexmodel_Notforward_200_SSIM_MSE_LeakyReLU_29042022', # two-branch SSIM + angle MSE
        'complexmodel_Notforward_200_ComplexMSE_LeakyReLU_13042022', # MSE
        'complexmodel_Notforward_200_SSIM_LeakyReLU_18042022', # envelope SSIM
        'complexmodel_Notforward_300_SSIM_LeakyReLU_22042022', # 2-branch SSIM filter size = 5
        'complexmodel_Notforward_300_SSIM_LeakyReLU_30042022' # 2-branch SSIM filter size = 7
        
        }
    for model_name in model_set:
        model = tf.keras.models.load_model(os.path.join(DIR_SAVED,model_name,model_name+'.h5'),
                                       custom_objects=get_custom_object())
        prediction = model.predict(np.expand_dims(test_img,axis=0))
        phase_diff(np.expand_dims(test_img,axis=0), prediction)
        show_fig(prediction,DR=60)
        del model
    
        
# class PostProcessing():
    
#     def __init__(self,
#                  model,
#                  name=None,
#                  DR=None,
#                  DIR_SAVED=r'./modelinfo',
#                  DIR_SIMULATION=r'./simulation_straight'):
#         self.model = model
#         self.name = name
#         self.DR = DR
        
#         self.DIR_SAVED = DIR_SAVED
#         self.DIR_SIMULATION = DIR_SIMULATION
        
#     def show_fig(self, img, ind=None, title_=None, saved=False):
        
#         # check image dimension is 3 or 4
#         if len(img.shape) == 4:
#             if img.shape[0] != 1:
#                 raise ValueError('unable to support multiple images')
#             shape = img.shape[1:-1]
#         elif len(img.shape) == 3:
#             shape = img.shape[:-1]
#         else:
#             raise ValueError('invalid image shape')
#         if ind is None:
#             axis = None
#         else:
#             axis = self._get_axis(img, ind)
#         plt.figure(dpi=300)
#         envelope_img = self.envelope_detection(img, self.DR)
#         envelope_img = envelope_img.reshape(shape)
#         assert len(envelope_img.shape) == 2
#         if self.DR is not None:
#             plt.imshow(envelope_img,
#                        cmap='gray',
#                        vmin=0,vmax=self.DR,
#                        extent=axis,
#                        aspect='auto')
#             plt.colorbar()
#         else:
#             plt.imshow(envelope_img, cmap='gray',extent=axis,aspect='auto')
#         if title_ is not None:
#             assert isinstance(title_, str)
#             plt.title(title_)
#         plt.xlabel('Lateral position (mm)')
#         plt.ylabel('Depth (mm)')
#         if saved:
#             saved_name = os.path.join(self.DIR_SAVED, self.name , title_ + '.png')
#             plt.savefig(saved_name, dpi=300)
#         plt.show()
      
#     def envelope_detection(self, signal):
        
#         '''
#         Parameters
#         ----------
#         signal : float32, numpy array
#         DR : int, dynamic range. The default is None.
#         Returns
#         -------
#         envelop-detected signal in float32. Its shape is same as input shape.
#         '''
#         # check dtype
#         if not np.isreal(signal).all():
#             raise ValueError('signal must be an one-channel or two-channel real-valued array')
#         # check rank
#         if len(signal.shape) == 3:
#             signal = np.expand_dims(signal, axis=0)
#         shape = signal.shape
#         assert len(shape) == 4
        
#         if shape[-1] == 2:
#             envelope = np.sqrt(signal[:,:,:,0]**2 + signal[:,:,:,1]**2)
#         elif shape[-1] == 1:
#             envelope = np.abs(Signal.hilbert(signal, axis=1))
#         else:
#             raise ValueError('signal channels must be one or two')
#         envelope = envelope.reshape(shape[:-1])
#         ratio = np.max(envelope, axis=(1,2),keepdims=True)
#         envelope = envelope/ratio
#         if self.DR is None:
#             return envelope
#         else:
#             dB_img = 20*np.log10(envelope + 1e-16) + self.DR
#             return dB_img
          
#     def save_model(self, history):
        
#         saved_dir = os.path.join(self.DIR_SAVED, self.name)
#         model_name = self.name + '.h5'      # model
#         model_arcname = self.name + 'arc.png'  # model architecture
#         if not os.path.exists(saved_dir):
#             try:
#                 os.mkdir(saved_dir)
#             except FileNotFoundError:
#                 os.makedirs(saved_dir)
#         model_saved_path = os.path.join(saved_dir, model_name)
#         model_arcpath = os.path.join(saved_dir, model_arcname)
#         # plot and save model architecture
#         tf.keras.utils.plot_model(self.model, to_file=model_arcpath, show_shapes=True, show_layer_names=True, dpi=900)
#         self.model.save(model_saved_path)
#         history_name = os.path.join(saved_dir, self.name + 'history.txt') # loss value per epoch
#         with open(history_name, 'w') as f:
#             f.write(str(history))
#         plt.figure()
#         plt.plot(history['loss'], label='training loss')
#         try:
#             plt.plot(history['val_loss'], label='val loss')
#         finally:
#             plt.legend()
#             plt.xlabel('epochs')
#             plt.ylabel('loss')
#             plt.savefig(os.path.join(saved_dir,self.name+'.png'))
#             plt.show()
            
#     def inference(self, testset):
        
#         time_collection = []
#         for i in range(10):
#             _ = self.model.predict(testset[i:i+10])
#         for i in range(10):
#             s = time.time()
#             _ = self.model.predict(testset)
#             e = time.time()
#             time_collection.append(e-s)
#         return np.mean(time_collection)/testset.shape[0]
    
#     def _get_axis(self, img, ind):
        
#         assert len(img.shape) ==3
#         r, c = img.shape[:2]
#         if (ind+1)%4 == 0:
#             level = 4
#         else:
#             level = (ind+1)%4
#         file_name = 'Data_' + str((ind+1)//4 + 1) + '_delay_' + str(level) + '.mat'
#         file_path = os.path.join(self.DIR_SIMULATION, file_name)
#         data = io.loadmat(file_path)
#         dx = data.get('dx') * (513/img.shape[1])
#         dz = data.get('dz') * (513/img.shape[0])
#         depth = data.get('depth')/2
#         x_axis = np.linspace(0,dx*c-dx,c) * 1e3 # [mm]
#         z_axis = np.linspace(0,dz*r-dz,r) * 1e3 + depth * 1e3 # [mm]
#         xmin, xmax = (np.min(x_axis), np.max(x_axis))
#         zmin, zmax = (np.min(z_axis), np.max(z_axis))
#         return (xmin, xmax, zmax, zmin)