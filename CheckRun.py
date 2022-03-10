# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 18:40:31 2022

@author: benzener
"""


import complexnn
import numpy as np
import tensorflow as tf
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)
        
x_train = np.random.randn(1,512,512,2)
y_train = np.random.randn(1,512,512,2)
tf.keras.backend.clear_session()
input_shapes = x_train.shape[1:]

# model = tf.keras.models.Sequential()
# model.add(complexnn.conv_test.ComplexConv2D(32, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 4)))
# model.add(complexnn.bn_test.ComplexBatchNormalization())
# model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
# model.built
# model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
# model.fit(x_train,y_train,epochs=2)

#-----test------
InputTensor = tf.keras.Input(shape=input_shapes)
conv1 = complexnn.conv_test.ComplexConv2D(64,(3,3),padding='same',strides=2)(InputTensor)
conv2 = complexnn.conv_test.ComplexConv2D(128,(3,3),padding='same',strides=2)(conv1)
conv2 = complexnn.bn_test.ComplexBatchNormalization()(conv2)
conv3 = complexnn.conv_test.ComplexConv2D(128,(3,3),padding='same',strides=2,transposed=True)(conv2)
Output= complexnn.conv_test.ComplexConv2D(64,(3,3),padding='same',strides=2,transposed=True)(conv3)
model = tf.keras.Model(inputs=InputTensor,outputs=Output)
model.compile(optimizer=tf.keras.optimizers.RMSprop(),loss=complexnn.loss.ComplexMSE)
model.build
model.fit(x_train,y_train, epochs=2)
model.summary()

#-----test------

# InputTensor = keras.Input(shape=input_shapes)
# conv1 = complexnn.conv.ComplexConv2D(64,(3,3),padding='same')(InputTensor)
# bn1 = complexnn.bn.ComplexBatchNormalization()(conv1)
# Output= complexnn.conv.ComplexConv2D(16,(3,3),padding='same')(bn1)
# model = keras.Model(inputs=InputTensor,outputs=Output)
# model.compile(optimizer=keras.optimizers.RMSprop(),loss='MSE')
# model.build
# model.fit(x_train,y_train,epochs=2)
# model.summary()