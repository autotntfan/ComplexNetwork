# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 22:13:05 2022

@author: benzener
"""
import tensorflow as tf
import csv
import os
from complexnn.activation import cReLU, zReLU, modReLU, AmplitudeMaxout, FLeakyReLU
from complexnn.loss import ComplexRMS, ComplexMAE, ComplexMSE, SSIM, MS_SSIM, SSIM_MSE
from complexnn.conv_test import ComplexConv2D
from complexnn.bn_test import ComplexBatchNormalization
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Dropout, Concatenate, LeakyReLU
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.activations import tanh
from datetime import datetime

class Model():
    
    def __init__(self,
                 filters=16,
                 size=(3,3),
                 batch_size=2,
                 lr=1e-4,
                 epochs=2,
                 validation_split=0.2,
                 seed=7414,
                 activations='LeakyReLU',
                 losses='ComplexMSE',
                 apply_batchnorm=True,
                 dropout_rate=None,
                 callbacks=None,
                 complex_network=True):
        self.filters = filters
        self.size = size
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.validation_rate = validation_split
        self.seed = seed
        self.activations = activations
        self.losses = losses
        self.apply_batchnorm = apply_batchnorm
        self.dropout_rate = dropout_rate
        self.callbacks = callbacks
        self.complex_network = complex_network
        
        self.input_shape = None
        self.forward = False
        
    
    def __call__(self, x, y):
        # check input is forward or not forward and get input_shape
        if isinstance(x,list):
            self.input_shape = x[0].shape[1:]
            self.forward = True
        else:
            self.input_shape = x.shape[1:]
        self.sanitized()          
        model = self.build_model()
        history = model.fit(x, y,
                            validation_split=self.validation_rate,
                            batch_size=self.batch_size,
                            verbose=2,
                            epochs=self.epochs,
                            callbacks=self.callbacks)
        return model, history
        
    def build_model(self):
        self.save_info()
        # determine activation function
        if self.activations in {'cReLU', 'zReLU', 'modReLU', 'LeakyReLU', 'AMU', 'FLeakyReLU'}:
            self.activations = {
                'cReLU': cReLU,
                'zReLU': zReLU,
                'modReLU': modReLU,
                'LeakyReLU': LeakyReLU,
                'FLeakyReLU': FLeakyReLU,
                'AMU'      : AmplitudeMaxout
                }[self.activations]
        else:
            if isinstance(self.activations, str):
                raise KeyError('activation function is not defined')

        
        # determine loss function 
        if self.losses in {'ComplexRMS', 'ComplexMAE', 'ComplexMSE', 'SSIM', 'MSE','MS_SSIM' ,'SSIM_MSE'}:
            self.losses = {
                'ComplexRMS': ComplexRMS,
                'ComplexMAE': ComplexMAE,
                'ComplexMSE': ComplexMSE,
                'SSIM'      : SSIM,
                'MS_SSIM'   : MS_SSIM,
                'SSIM_MSE'  : SSIM_MSE,
                'MSE'       : MeanSquaredError()
                }[self.losses]
            
        self.convFunc = ComplexConv2D if self.complex_network else Conv2D
        self.bnFunc = ComplexBatchNormalization if self.complex_network else BatchNormalization
        
        model = self.UNet()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss=self.losses)
        model.summary()
        return model
            
    def downsample(self, filters, size):
        
        result = tf.keras.Sequential()

        result.add(
            self.convFunc(filters, size, strides=2, padding='same', use_bias=False)
            )
        if self.apply_batchnorm:
            result.add(self.bnFunc())
        
        result.add(self.activations())
        result.add(self.convFunc(filters, size, padding='same', use_bias=False))
        if self.apply_batchnorm:
            result.add(self.bnFunc())
        result.add(self.activations())
        return result

    def upsample(self, filters, size):
        
        result = tf.keras.Sequential()
        
        if self.complex_network:
            result.add(self.convFunc(filters, size, strides=2, padding='same', transposed=True))
        else:
            result.add(Conv2DTranspose(filters, size, strides=2, padding='same'))
        result.add(self.bnFunc())
        if self.dropout_rate:
            result.add(Dropout(self.dropout_rate))
        result.add(self.activations())
        result.add(self.convFunc(filters, size, padding='same', use_bias=False))
        result.add(self.bnFunc())
        result.add(self.activations())
        return result
        
    def UNet(self):
        tf.random.set_seed(self.seed)
        inputs = Input(self.input_shape)
        if self.forward:
            inputs_forward = Input(self.input_shape)
        down_stack = [
            self.downsample(4*self.filters, self.size), #(bs, 128, 128, 32*2)
            self.downsample(8*self.filters, self.size),  #(bs, 64, 64, 64*2)
            self.downsample(16*self.filters, self.size),  #(bs, 32, 32, 128*2)
            self.downsample(32*self.filters, self.size),  #(bs, 16, 16, 256*2)
            self.downsample(64*self.filters, self.size), #(bs, 8, 8, 512*2)
            self.downsample(64*self.filters, self.size), #(bs, 4, 4, 512*2)
            ]
        up_stack = [
            self.upsample(64*self.filters, self.size), #(bs, 8, 8, 512*2)
            self.upsample(32*self.filters, self.size), #(bs, 16, 16, 512*2)
            self.upsample(16*self.filters, self.size),  #(bs, 32, 32, 256*2)
            self.upsample(8*self.filters, self.size),  #(bs, 64, 64, 128*2)
            self.upsample(4*self.filters, self.size),  #(bs, 128, 128, 64*2)
            ]
        if self.complex_network:
            last = self.convFunc(2*self.filters, self.size, strides=2, padding='same', transposed=True) #(bs, 256, 256, 16*2)
        else:
            last = Conv2DTranspose(2*self.filters, self.size, strides=2, padding='same')
        x = inputs
    
        
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)
        
        skips = reversed(skips[:-1])
        
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = Concatenate()([x, skip])
        x = last(x)
        x = self.activations()(x)
          
        for _ in range(3):
            x = self.convFunc(2*self.filters, 3, padding='same', use_bias=False)(x)
            x = self.bnFunc()(x)
            x = self.activations()(x)
        x = self.convFunc(1, 3, padding='same', use_bias=False)(x)
        x = self.bnFunc()(x)
    
        
        if self.forward:
            x = self.activations()(x)
            x = Concatenate(axis=-1)((x, inputs_forward))
            x = self.convFunc(2, 3, padding='same', use_bias=False)(x)
            x = self.bnFunc()(x)
            x = tanh(x)
            return tf.keras.Model(inputs=[inputs, inputs_forward], outputs=x)
        else:    
            x = tf.keras.layers.LeakyReLU()(x)
            return tf.keras.Model(inputs=inputs, outputs=x)

    def sanitized(self):
        if self.complex_network:
            assert self.input_shape[-1] == 2
            if self.losses not in {'ComplexRMS', 'ComplexMAE', 'ComplexMSE','SSIM','MS_SSIM','SSIM_MSE'}:
                raise KeyError('Invalid complex-valued loss function')
            if self.activations not in {'cReLU', 'zReLU', 'modReLU', 'LeakyReLU', 'AMU', 'FLeakyReLU'}:
                raise  KeyError('Unsupported activation')
        else:
            assert self.input_shape[-1] == 1
            if self.losses not in {'MSE','SSIM', 'MS_SSIM', 'SSIM_MSE'}:
                raise KeyError('Invalid real-valued loss function')
            if self.activations not in {'LeakyReLU'}:
                raise  KeyError('Unsupported activation')
    
    def generate_name(self):
        now = datetime.now()
        day_month_year = now.strftime("%d%m%Y")
        type_ = 'complex' if self.complex_network else 'real'
        forward = 'forward' if self.forward else 'Notforward'
        epochs = str(self.epochs)
        return f'{type_}model_{forward}_{epochs}_{self.losses}_{self.activations}_' + day_month_year
    
    def save_info(self):
        saved_var = {
            'input_shape':self.input_shape,
            'forward':self.forward,
            'callback':self.callbacks,
            'complex':self.complex_network,
            'validation_split':self.validation_rate,
            'filters':self.filters,
            'kernel_size':self.size,
            'learning_rate':self.lr,
            'batch_size':self.batch_size,
            'epochs':self.epochs,
            'activation':self.activations,
            'loss':self.losses,
            }
        file_name = self.generate_name()
        saved_dir = os.path.join(r'./modelinfo', file_name)
        file_name = file_name + '_parameters.txt'
        if not os.path.exists(saved_dir):
            try:
                os.mkdir(saved_dir)
            except FileNotFoundError:
                os.makedirs(saved_dir)
        saved_path = os.path.join(saved_dir, file_name)
        with open(saved_path, 'w') as f:
            f.write(str(saved_var))
        
            
            
        



        
        
        
        
        