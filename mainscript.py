# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 00:17:18 2022

@author: benzener
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import complexnn
import h5py
import numpy as np
import scipy
import os
import multiprocessing

def read_data(dtype='psf'):
    data = None
    if dtype == 'psf':
        type_ = 'psf_bb'
    elif dtype == 'speckle':
        type_ = 'speckle_bb'
    else:
        raise KeyError('dtype is invalid')
    with h5py.File('D:\ComplxDataset\dataset.hdf5','r') as f:
        for level in range(1,5):
            key = 'level' + str(level) + '/' + type_
            img = f[key][:].astype(np.float32)[:,1:,1:,:] # original 513*513 -> 512*512
            if data is None:
                data = img
            else:
                data = np.vstack((data,img))
                break
    return data

def normalize_complex_value(x):
    if len(x.shape) == 3:
        x = np.expand_dims(x,axis=0)
    assert len(x.shape) == 4
    channel = x.shape[-1]//2
    real = x[:,:,:,:channel]
    imag = x[:,:,:,channel:]
    modulus = np.sqrt(real**2 + imag**2)
    factor = np.max(modulus,axis=(1,2,3)).reshape(x.shape[0],1,1,1)
    return np.nan_to_num(x/factor)
def reduce_sampling_rate(signal, factor=2):
    return signal[:,::factor,::factor,:]

def check_data_range(x):
    max_ = np.max(x)
    min_ = np.min(x)
    print(f'the largest value is {max_} and the smallest one is {min_}')
    
def downsample(filters, size, apply_batchnorm=False):
    result = tf.keras.Sequential()
    result.add(
        complexnn.conv_test.ComplexConv2D(filters, size, strides=2, padding='same', use_bias=False)
        )
    if apply_batchnorm:
        result.add(complexnn.bn_test.ComplexBatchNormalization())
    
    result.add(complexnn.activation.cLeakyReLU())
    result.add(complexnn.conv_test.ComplexConv2D(filters, 3, padding='same', use_bias=False))
    if apply_batchnorm:
        result.add(complexnn.bn_test.ComplexBatchNormalization())
    result.add(complexnn.activation.cLeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    result = tf.keras.Sequential()
    result.add(
        complexnn.conv_test.ComplexConv2D(filters, size,strides=2, padding='same',
                                          transposed=True)
        )
    result.add(complexnn.bn_test.ComplexBatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(complexnn.activation.cLeakyReLU())
    result.add(complexnn.conv_test.ComplexConv2D(filters, 3, padding='same', use_bias=False))
    result.add(complexnn.bn_test.ComplexBatchNormalization())
    result.add(complexnn.activation.cLeakyReLU())
    return result

def UNet(OUTPUT_CHANNELS=1, input_size=(512,512,2), forward=False):
    inputs = tf.keras.layers.Input(input_size)
    if forward:
        inputs_forward = tf.keras.layers.Input(input_size)
    down_stack = [
        downsample(32, 3, apply_batchnorm=True), #(bs, 128, 128, 32*2)
        downsample(64, 3),  #(bs, 64, 64, 64*2)
        downsample(128, 3),  #(bs, 32, 32, 128*2)
        downsample(256, 3),  #(bs, 16, 16, 256*2)
        downsample(512, 3), #(bs, 8, 8, 512*2)
        ]
    up_stack = [
        upsample(512, 3), #(bs, 16, 16, 512*2)
        upsample(256, 3),  #(bs, 32, 32, 256*2)
        upsample(128, 3),  #(bs, 64, 64, 128*2)
        upsample(64, 3),  #(bs, 128, 128, 64*2)
        ]
    last = complexnn.conv_test.ComplexConv2D(32, 3, strides=2, padding='same',
                                      activation='cLeakyReLU',
                                      transposed=True) #(bs, 256, 256, 1*2)
    x = inputs

    
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    skips = reversed(skips[:-1])
    
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    x = last(x)
      
    for _ in range(3):
        x = complexnn.conv_test.ComplexConv2D(16, 3, padding='same', use_bias=False)(x)
        x = complexnn.bn_test.ComplexBatchNormalization()(x)
        x = complexnn.activation.cLeakyReLU()(x)
    x = complexnn.conv_test.ComplexConv2D(OUTPUT_CHANNELS, 3, padding='same', use_bias=False)(x)
    x = complexnn.bn_test.ComplexBatchNormalization()(x)
    # x = tf.keras.activations.tanh(x)

    
    if forward:
        x = complexnn.activation.cLeakyReLU()(x)
        x = tf.keras.layers.Concatenate(axis=-1)((x, inputs_forward))
        x = complexnn.conv_test.ComplexConv2D(OUTPUT_CHANNELS, 3, padding='same', use_bias=False)(x)
        x = complexnn.bn_test.ComplexBatchNormalization()(x)
        x = complexnn.activation.ctanh()(x)
        return tf.keras.Model(inputs=[inputs, inputs_forward], outputs=x)
    else:    
        x = complexnn.activation.ctanh()(x)
        return tf.keras.Model(inputs=inputs, outputs=x)

def envelope_detection(signal, DR=None):
    if not np.isreal(signal).all():
        raise TypeError('signal must be an one-channel or two-channel real-valued array')
    
    if signal.shape[-1] == 2:
        envelope = np.sqrt(signal[:,:,0]**2 + signal[:,:,1]**2)
    elif signal.shape[-1] == 1 and np.isreal(signal).all():
        envelope = np.abs(scipy.signal.hilbert(signal))
    else:
        raise TypeError(f'Ivalid shape {signal.shape} for signal')
        
    if DR is None:
        return envelope
    else:
        dB_img = 20*np.log10(envelope/np.max(envelope)+1e-16) + DR
        return dB_img
        
if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    psf = normalize_complex_value(read_data(dtype='psf'))
    speckle = normalize_complex_value(read_data(dtype='speckle'))
    indices = np.arange(400)
    np.random.shuffle(indices)
    x_train = speckle[indices[:300]]
    y_train = psf[indices[:300]]
    x_test = speckle[indices[300:]]
    y_test = psf[indices[300:]]
    x_train = reduce_sampling_rate(x_train)
    y_train = reduce_sampling_rate(y_train)
    
    plt.figure(dpi=300)
    plt.imshow(envelope_detection(x_train[0]), cmap='gray')
    plt.colorbar()
    plt.figure(dpi=300)
    plt.imshow(envelope_detection(y_train[0]), cmap='gray')
    plt.colorbar()
    def run_tensorflow(x, y):
        shape = x.shape[1:]
        model = UNet(input_size=shape)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=complexnn.loss.ComplexMSE)
        model.summary()
        model.fit(x, y, batch_size=2, verbose=2, epochs=10)
        return model
    model = run_tensorflow(x_train,y_train)
    prediction = model.predict(x_test[:1,::2,::2,:])
    plt.figure(dpi=300)
    plt.imshow(envelope_detection(x_test[0,::2,::2,:]), cmap='gray')
    plt.colorbar()
    plt.figure(dpi=300)
    plt.imshow(envelope_detection(prediction[0]), cmap='gray')
    plt.colorbar()
    tf.keras.backend.clear_session()
    check_data_range(prediction[0])
    

    
        