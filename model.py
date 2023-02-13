# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 22:13:05 2022

@author: benzener
"""
import tensorflow as tf
import os
import numpy as np
import time
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
                 filters=4,
                 size=(3,3),
                 batch_size=2,
                 lr=1e-4,
                 epochs=8,
                 validation_split=0.2,
                 validation_data=None,
                 seed=7414,
                 activations='LeakyReLU',
                 losses='ComplexMSE',
                 forward=False,
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
        self.validation_data = validation_data
        self.seed = seed
        self.activations = activations
        self.losses = losses
        self.apply_batchnorm = apply_batchnorm
        self.dropout_rate = dropout_rate
        self.callbacks = callbacks
        self.complex_network = complex_network
        
        self.forward = forward
        
    
    def __call__(self, x, y):
        # check input is forward or not forward and get input_shape
        if isinstance(x,list):
            input_shape = x[0].shape[1:]
            self.forward = True
        else:
            input_shape = x.shape[1:]
        model = self.build_model(input_shape)
        # lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay([50,100],
                                                                            # [self.lr,0.5*self.lr,0.1*self.lr])
        # model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9), loss=self.losses)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss=self.losses)
        # model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=self.lr), loss=self.losses)
        model.summary()
        history = model.fit(x, y,
                            validation_split=self.validation_rate,
                            validation_data=self.validation_data,
                            batch_size=self.batch_size,
                            verbose=2,
                            epochs=self.epochs,
                            callbacks=self.callbacks)
        return model, history
        
    def build_model(self, input_shape):
        self.input_shape = input_shape
        self.sanitized()
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
        return self.core()
            
    def downsample(self, filters, size):
        
        result = tf.keras.Sequential()

        result.add(
            self.convFunc(filters, size, strides=2, padding='same', use_bias=False)
            )
        if self.apply_batchnorm:
            result.add(self.bnFunc())
        if self.dropout_rate:
            result.add(Dropout(self.dropout_rate))
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
        if self.apply_batchnorm:
            result.add(self.bnFunc())
        if self.dropout_rate:
            result.add(Dropout(self.dropout_rate))
        result.add(self.activations())
        result.add(self.convFunc(filters, size, padding='same', use_bias=False))
        result.add(self.bnFunc())
        result.add(self.activations())
        return result
        
    def core(self):
        tf.random.set_seed(self.seed)
        inputs = Input(self.input_shape)
        if self.forward:
            inputs_forward = Input(self.input_shape)
        down_stack = [
            self.downsample(4*self.filters, self.size), #(bs, 64, 128, 32*2)
            self.downsample(8*self.filters, self.size),  #(bs, 32, 64, 64*2)
            self.downsample(16*self.filters, self.size),  #(bs, 16, 32, 128*2)
            self.downsample(32*self.filters, self.size),  #(bs, 8, 16, 256*2)
            self.downsample(64*self.filters, self.size), #(bs, 4, 8, 512*2)
            self.downsample(64*self.filters, self.size), #(bs, 2, 4, 512*2)
            ]
        up_stack = [
            self.upsample(64*self.filters, self.size), #(bs, 4, 8, 512*2)
            self.upsample(32*self.filters, self.size), #(bs, 8, 16, 512*2)
            self.upsample(16*self.filters, self.size),  #(bs, 16, 32, 256*2)
            self.upsample(8*self.filters, self.size),  #(bs, 32, 64, 128*2)
            self.upsample(4*self.filters, self.size),  #(bs, 64, 128, 64*2)
            ]
        if self.complex_network:
            last = self.convFunc(2*self.filters, self.size, strides=2, padding='same', transposed=True) #(bs, 128, 256, 16*2)
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
            x = tf.keras.layers.LeakyReLU()(x)
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
            if self.activations not in {'LeakyReLU','FLeakyReLU'}:
                raise  KeyError('Unsupported activation')
    
    def generate_name(self):
        now = datetime.now()
        day_month_year = now.strftime("%d%m%Y")
        type_ = 'complex' if self.complex_network else 'real'
        forward = 'forward' if self.forward else 'Notforward'
        epochs = str(self.epochs)
        return f'{type_}model_{forward}_{epochs}_{self.losses}_{self.activations}_' + day_month_year
    
    def info(self):
        return {
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
            'activation':str(self.activations),
            'loss':str(self.losses),
            }


            
    def running(self, x, y):
        '''
            without supporting Forward model
        '''
        if self.validation_rate:
            num_train = round(x.shape[0]*(1-self.validation_rate))
            x_train, y_train = x[:num_train], y[:num_train]
            x_val, y_val = x[num_train:], y[num_train:]
        if self.validation_data:
            x_train, y_train = x, y
            x_val, y_val = self.validation_data
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(self.batch_size)
        valid_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(self.batch_size)
        self.model = self.build_model(x_train.shape[1:])
        self.model.summary()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        history = {
            'loss':[],
            'val_loss':[]
            }
        for epoch in range(self.epochs):
            s = time.time()
            loss_train_epoch = []
            loss_valid_epoch = []
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                loss_train_epoch.append(self.train_step(x_batch_train, y_batch_train))
            history['loss'].append(np.mean(loss_train_epoch))
            for x_batch_val, y_batch_val in valid_dataset:
                loss_valid_epoch.append(self.test_step(x_batch_val, y_batch_val))
            history['val_loss'].append(np.mean(loss_valid_epoch))
            e = time.time()
            print(f'Epoch:{epoch+1}/{self.epochs} - {(e-s):.1f}s - ' \
                  f'loss:{np.mean(loss_train_epoch):.6f} - ' \
                  f'val_loss:{np.mean(loss_valid_epoch):.6f} \n')
        return self.model, history
    
    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            result = self.model(x, training=True)
            loss_value = self.losses(y, result)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss_value
    
    @tf.function
    def test_step(self, x, y):
        val_result = self.model(x, training=False)
        return self.losses(y, val_result)