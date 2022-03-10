#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 14:47:57 2020

@author: Wei-Hsiang, Shen
"""

import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt


from dataset import GetDS

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())
  
  # another conv layer
  result.add(tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False))
  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())
  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())
  
  # another conv layer
  result.add(tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False))
  result.add(tf.keras.layers.BatchNormalization())
  result.add(tf.keras.layers.LeakyReLU())

  return result

def UNet(INPUT_CHANNELS = 1, OUTPUT_CHANNELS = 1, input_size=(256,256)):
  inputs = tf.keras.layers.Input(shape=[input_size[0], input_size[1], INPUT_CHANNELS])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, 4), # (bs, 64, 64, 128)
    downsample(256, 4), # (bs, 32, 32, 256)
    downsample(512, 4), # (bs, 16, 16, 512)
    downsample(1024, 4), # (bs, 8, 8, 512)
    downsample(1024, 4), # (bs, 4, 4, 512)
    # downsample(512, 4), # (bs, 2, 2, 512)
    # downsample(512, 4), # (bs, 1, 1, 512)
  ]

  up_stack = [
    # upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    # upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(1024, 4), # (bs, 8, 8, 1024)
    upsample(512, 4), # (bs, 16, 16, 1024)
    upsample(256, 4), # (bs, 32, 32, 512)
    upsample(128, 4), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(32, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

  x = inputs
  
  if input_size[0] == 192*2:
      x = downsample(32, 4)(x)

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)
  
  for _ in range(3):
      x = tf.keras.layers.Conv2D(32, 3, padding='same', use_bias=False)(x)
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.LeakyReLU()(x)
  
  x = tf.keras.layers.Conv2D(OUTPUT_CHANNELS, 3, padding='same', use_bias=False)(x)
  x = tf.keras.layers.BatchNormalization()(x)
  # x = tf.keras.activations.tanh(x)
  x = tf.keras.layers.LeakyReLU()(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

def UNetF(INPUT_CHANNELS = 1, OUTPUT_CHANNELS = 1, input_size=(256,256)):
  inputs = tf.keras.layers.Input(shape=[input_size[0], input_size[1], INPUT_CHANNELS])
  inputs_forward = tf.keras.layers.Input(shape=[input_size[0], input_size[1], INPUT_CHANNELS])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, 4), # (bs, 64, 64, 128)
    downsample(256, 4), # (bs, 32, 32, 256)
    downsample(512, 4), # (bs, 16, 16, 512)
    downsample(1024, 4), # (bs, 8, 8, 512)
    downsample(1024, 4), # (bs, 4, 4, 512)
    # downsample(512, 4), # (bs, 2, 2, 512)
    # downsample(512, 4), # (bs, 1, 1, 512)
  ]

  up_stack = [
    # upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    # upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(1024, 4), # (bs, 8, 8, 1024)
    upsample(512, 4), # (bs, 16, 16, 1024)
    upsample(256, 4), # (bs, 32, 32, 512)
    upsample(128, 4), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(32, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

  x = inputs
  
  if input_size[0] == 192*2:
      x = downsample(32, 4)(x)

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)
  
  for _ in range(3):
      x = tf.keras.layers.Conv2D(32, 3, padding='same', use_bias=False)(x)
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.LeakyReLU()(x)
      
  x = tf.keras.layers.Conv2D(2, 3, padding='same', use_bias=False)(x)
  x = tf.keras.layers.BatchNormalization()(x)
  # x = tf.keras.activations.tanh(x)
  x = tf.keras.layers.LeakyReLU()(x)
  
  # forward input
  x = tf.keras.layers.Concatenate(axis=-1)((x, inputs_forward))
  x = tf.keras.layers.Conv2D(OUTPUT_CHANNELS, 3, padding='same', use_bias=False)(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.LeakyReLU()(x)

  return tf.keras.Model(inputs=[inputs, inputs_forward], outputs=x)

if __name__ == '__main__':
    # %% standard UNet
    model = UNet()
    tf.keras.utils.plot_model(model, show_shapes=True, dpi=64)
    
    train_ds, val_ds = GetDS()
    
    for patch, psf in train_ds.take(1):
        pred = model(patch)
    
    plt.figure(dpi=300)
    plt.imshow(patch[0,:,:,0], cmap='gray')
    plt.colorbar()
    
    plt.figure(dpi=300)
    plt.imshow(psf[0,:,:,0], cmap='gray')
    plt.colorbar()
    
    plt.figure(dpi=300)
    plt.imshow(pred[0,:,:,0], cmap='gray')
    plt.colorbar()
    
    # %% Forward UNet
    model = UNetF()
    
    train_ds, val_ds = GetDS(forward_flag=True)
    
    for patch, psf in train_ds.take(1):
        pred = model(patch)
    
    plt.figure(dpi=300)
    plt.imshow(patch[0][0,:,:,0], cmap='gray')
    plt.colorbar()
    
    plt.figure(dpi=300)
    plt.imshow(patch[1][0,:,:,0], cmap='gray')
    plt.colorbar()
    
    plt.figure(dpi=300)
    plt.imshow(psf[0,:,:,0], cmap='gray')
    plt.colorbar()
    
    plt.figure(dpi=300)
    plt.imshow(pred[0,:,:,0], cmap='gray')
    plt.colorbar()