# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 00:38:12 2022

@author: benzener
"""
import complexnn
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer,MaxPooling2D



#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
#    try:
#        for gpu in gpus:
#            tf.config.experimental.set_memory_growth(gpu, True)
#
#    except RuntimeError as e:
#        print(e)
#        
np.random.seed(1)
x_train = np.random.rand(61,64,64,4)
y_train = np.random.rand(61,64,64,2)


tf.keras.backend.clear_session()
input_shapes = x_train.shape[1:]

#InputTensor = keras.Input(shape=input_shapes)
#conv1 = complexnn.conv.ComplexConv2D(64,(3,3),padding='same')(InputTensor)
#bn1 = complexnn.bn.ComplexBatchNormalization()(conv1)
#amp1 = complexnn.activation.modReLU()(conv1)
#conv2 = complexnn.conv.ComplexConv2D(16,(3,3),padding='same',activation='zReLU')(amp1)
#amp2 = complexnn.activation.AmplitudeMaxout(4)(conv2)
#Output = complexnn.conv.ComplexConv2D(1,(1,1),padding='same',activation='zReLU')(amp2)
#model = keras.Model(inputs=InputTensor,outputs=Output)
#model.compile(optimizer=keras.optimizers.RMSprop(),loss=complexnn.loss.ComplexMSE)
#model.build
#model.fit(x_train,y_train,epochs=2)

#InputTensor = keras.Input(shape=input_shapes)
#Output = complexnn.conv.ComplexConv2D(64,(3,3),padding='same',activation='relu')(InputTensor)
##Output = complexnn.bn.ComplexBatchNormalization()(conv1)
#model = keras.Model(inputs=InputTensor,outputs=Output)
#model.compile(optimizer=keras.optimizers.RMSprop(),loss=complexnn.loss.ComplexMSE)
#model.build
#model.fit(x_train,y_train,epochs=2)

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0, 0.02)
    result = Sequential()
    result.add(complexnn.conv.ComplexConv2D(filters, size, strides=2, padding='same',
                                            kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(complexnn.bn.ComplexBatchNormalization())
    result.add(complexnn.activation.zReLU())

model = Sequential()

model.add(complexnn.conv.ComplexConv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 2)))
model.add(complexnn.bn.ComplexBatchNormalization())
model.add(MaxPooling2D((2, 2), padding='same'))

model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
