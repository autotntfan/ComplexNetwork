# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 00:17:18 2022

@author: benzener
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import complexnn
import numpy as np
import scipy
import os
from dataset import DataPreprocessing

    

def check_data_range(x):
    max_ = np.max(x)
    min_ = np.min(x)
    print(f'the largest value is {max_} and the smallest one is {min_}')

def show_fig(img, title_=None):
    plt.figure(dpi=300)
    plt.imshow(envelope_detection(img), cmap='gray')
    plt.title(title_)
    if title_ is not None:
        assert isinstance(title_, str)
        plt.title(title_)
    plt.show()


    
def downsample(filters, size, apply_batchnorm=False):
    result = tf.keras.Sequential()
    result.add(
        complexnn.conv_test.ComplexConv2D(filters, size, strides=2, padding='same', use_bias=False)
        )
    if apply_batchnorm:
        result.add(complexnn.bn_test.ComplexBatchNormalization())
    
    result.add(complexnn.activation.cLeakyReLU())
    result.add(complexnn.conv_test.ComplexConv2D(filters, size, padding='same', use_bias=False))
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
    result.add(complexnn.conv_test.ComplexConv2D(filters, size, padding='same', use_bias=False))
    result.add(complexnn.bn_test.ComplexBatchNormalization())
    result.add(complexnn.activation.cLeakyReLU())
    return result

def UNet(OUTPUT_CHANNELS=1, input_size=(512,512,2), forward=False):
    inputs = tf.keras.layers.Input(input_size)
    if forward:
        inputs_forward = tf.keras.layers.Input(input_size)
    down_stack = [
        downsample(64, 3, apply_batchnorm=True), #(bs, 128, 128, 32*2)
        downsample(128, 3),  #(bs, 64, 64, 64*2)
        downsample(256, 3),  #(bs, 32, 32, 128*2)
        downsample(512, 3),  #(bs, 16, 16, 256*2)
        downsample(512, 3), #(bs, 8, 8, 512*2)
        ]
    up_stack = [
        upsample(512, 3), #(bs, 8, 8, 512*2)
        upsample(256, 3),  #(bs, 16, 16, 256*2)
        upsample(128, 3),  #(bs, 32, 32, 128*2)
        upsample(64, 3),  #(bs, 64, 64, 64*2)
        ]
    last = complexnn.conv_test.ComplexConv2D(32, 3, strides=2, padding='same',
                                      activation='modReLU',
                                      transposed=True) #(bs, 256, 256, 16*2)
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
        x = complexnn.conv_test.ComplexConv2D(32, 3, padding='same', use_bias=False)(x)
        x = complexnn.bn_test.ComplexBatchNormalization()(x)
        x = complexnn.activation.cLeakyReLU()(x)
    x = complexnn.conv_test.ComplexConv2D(OUTPUT_CHANNELS, 3, padding='same', use_bias=False)(x)
    x = complexnn.bn_test.ComplexBatchNormalization()(x)

    
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
    # check rank
    if len(signal.shape) == 4: 
        signal = signal.reshape(signal.shape[1:])
    assert len(signal.shape) == 3
    
    # check dtype
    if not np.isreal(signal).all():
        raise TypeError('signal must be an one-channel or two-channel real-valued array')
        
    # check last dimension
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
    
def run_tensorflow(x, y, foward=False):
    if isinstance(x,list):
        shape = x[0].shape[1:]
    else:
        shape = x.shape[1:]
    model = UNet(input_size=shape, forward=forward)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=complexnn.loss.ComplexMSE)
    model.summary()
    history = model.fit(x, y, batch_size=4, verbose=2, epochs=1)
    return model, history
    
if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    
    forward = True
    preprocessing = DataPreprocessing()
    (x_train, y_train), (x_test, y_test), (ideal_train, ideal_test) = preprocessing.get_data()
    indices = preprocessing.indices
    custom_object = {
        'ComplexConv2D':complexnn.conv_test.ComplexConv2D,
        'ComplexBatchNormalization':complexnn.bn_test.ComplexBatchNormalization,
        'ComplexMSE':complexnn.loss.ComplexMSE
        }
    
    show_fig(x_train[0], 'speckle')
    show_fig(y_train[0], 'psf')
    if forward:
        # model, history = run_tensorflow([x_train, ideal_train], y_train, forward)
        model = tf.keras.models.load_model('modelF_bnT_100.h5',custom_objects=custom_object)
        prediction = model.predict([x_test, ideal_test])
    else:
        model, history = run_tensorflow(x_train, y_train)
        #model = tf.keras.models.load_model('model_bnT_200.h5',custom_objects=custom_object)
        prediction = model.predict(x_test)
    # plt.figure()
    # plt.plot(history.history['loss'])
    # plt.title('loss')
    # plt.show()
    
    num = 3
    show_fig(x_test[num], 'speckle')
    show_fig(prediction[num], 'predicted image')
    show_fig(y_test[num], 'psf')

    tf.keras.backend.clear_session()
    check_data_range(prediction[0])
    

    
        