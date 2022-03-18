# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 22:13:05 2022

@author: benzener
"""
import tensorflow as tf
import complexnn
import os
from complexnn.activation import cReLU, cLeakyReLU, ctanh, zReLU, modReLU
from complexnn.loss import ComplexRMS, ComplexMAE, ComplexMSE

class Model():
    
    def __init__(self,
                 filters=16,
                 size=3,
                 batch_size=2,
                 lr=1e-4,
                 epochs=2,
                 validation_split=0.2,
                 activations=None,
                 losses=None,
                 apply_batchnorm=True,
                 dropout_rate=None,
                 callbacks=None):
        self.filters = filters
        self.size = size
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.validation_rate = validation_split
        self.activations = activations
        self.losses = losses
        self.apply_batchnorm = apply_batchnorm
        self.dropout_rate = dropout_rate
        self.callbacks = callbacks
        
        self.input_shape = None
        self.forward = False
        self.input_shape = (256,256,2)
    
    def __call__(self, x, y):
        # check input is forward or not forward and get input_shape
        if isinstance(x,list):
            self.input_shape = x[0].shape[1:]
            self.forward = True
        else:
            self.input_shape = x.shape[1:]
        model = self.build_model()
        history = model.fit(x, y,
                            validation_split=self.validation_rate,
                            batch_size=self.batch_size,
                            verbose=2,
                            epochs=self.epochs,
                            callbacks=self.callbacks)
        return model, history
        
    def build_model(self):
        # determine activation function
        if self.activations in {'cReLU', 'cLeakyReLU', 'ctanh', 'zReLU', 'modReLU'}:
            self.activations = {
                'cReLU': cReLU,
                'cLeakyReLU': cLeakyReLU,
                'ctanh': ctanh,
                'zReLU': zReLU,
                'modReLU': modReLU}[self.activations]
        else:
            self.activations = cLeakyReLU
        
        # determine loss function 
        if self.losses in {'ComplexRMS', 'ComplexMAE', 'ComplexMSE'}:
            self.losses = {
                'ComplexRMS': ComplexRMS,
                'ComplexMAE': ComplexMAE,
                'ComplexMSE': ComplexMSE
                }[self.losses]
        else:
            self.losses = ComplexMSE
        
        model = self.UNet()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss=self.losses)
        model.summary()
        return model
            
    def downsample(self, filters, size):
        result = tf.keras.Sequential()
        result.add(
            complexnn.conv_test.ComplexConv2D(filters, size, strides=2, padding='same', use_bias=False)
            )
        if self.apply_batchnorm:
            result.add(complexnn.bn_test.ComplexBatchNormalization())
        
        result.add(self.activations())
        result.add(complexnn.conv_test.ComplexConv2D(filters, size, padding='same', use_bias=False))
        if self.apply_batchnorm:
            result.add(complexnn.bn_test.ComplexBatchNormalization())
        result.add(self.activations())
        return result

    def upsample(self, filters, size):
        result = tf.keras.Sequential()
        result.add(
            complexnn.conv_test.ComplexConv2D(filters, size, strides=2, padding='same',
                                              transposed=True)
            )
        result.add(complexnn.bn_test.ComplexBatchNormalization())
        if self.dropout_rate:
            result.add(tf.keras.layers.Dropout(self.dropout_rate))
        result.add(self.activations())
        result.add(complexnn.conv_test.ComplexConv2D(filters, size, padding='same', use_bias=False))
        result.add(complexnn.bn_test.ComplexBatchNormalization())
        result.add(self.activations())
        return result
        
    def UNet(self):
        inputs = tf.keras.layers.Input(self.input_shape)
        if self.forward:
            inputs_forward = tf.keras.layers.Input(self.input_shape)
        down_stack = [
            self.downsample(4*self.filters, self.size), #(bs, 128, 128, 32*2)
            self.downsample(8*self.filters, self.size),  #(bs, 64, 64, 64*2)
            self.downsample(16*self.filters, self.size),  #(bs, 32, 32, 128*2)
            self.downsample(32*self.filters, self.size),  #(bs, 16, 16, 256*2)
            self.downsample(64*self.filters, self.size), #(bs, 8, 8, 512*2)
            self.downsample(64*self.filters, self.size), #(bs, 8, 8, 512*2)
            ]
        up_stack = [
            self.upsample(64*self.filters, self.size), #(bs, 8, 8, 512*2)
            self.upsample(32*self.filters, self.size), #(bs, 8, 8, 512*2)
            self.upsample(16*self.filters, self.size),  #(bs, 16, 16, 256*2)
            self.upsample(8*self.filters, self.size),  #(bs, 32, 32, 128*2)
            self.upsample(4*self.filters, self.size),  #(bs, 64, 64, 64*2)
            ]
        last = complexnn.conv_test.ComplexConv2D(2*self.filters, self.size, strides=2, padding='same',
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
        x = self.activations()(x)
          
        for _ in range(3):
            x = complexnn.conv_test.ComplexConv2D(2*self.filters, self.size, padding='same', use_bias=False)(x)
            x = complexnn.bn_test.ComplexBatchNormalization()(x)
            x = self.activations()(x)
        x = complexnn.conv_test.ComplexConv2D(1, self.size, padding='same', use_bias=False)(x)
        x = complexnn.bn_test.ComplexBatchNormalization()(x)
    
        
        if self.forward:
            x = self.activations()(x)
            x = tf.keras.layers.Concatenate(axis=-1)((x, inputs_forward))
            x = complexnn.conv_test.ComplexConv2D(1, self.size, padding='same', use_bias=False)(x)
            x = complexnn.bn_test.ComplexBatchNormalization()(x)
            x = ctanh()(x)
            return tf.keras.Model(inputs=[inputs, inputs_forward], outputs=x)
        else:    
            x = ctanh()(x)
            return tf.keras.Model(inputs=inputs, outputs=x)



        
        
        
        
        