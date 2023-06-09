#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Authors: Olexa Bilaniuk
#

import tensorflow.keras.backend             as KB
import tensorflow.keras.layers              as KL
import tensorflow.keras.optimizers          as KO
import numpy                                as np
import tensorflow as tf
from tensorflow.python.keras.utils import conv_utils
from .utils_test import get_realpart, get_imagpart, get_abs
#
# Spectral Pooling Layer
#

class SpectralPooling1D(KL.Layer):
	def __init__(self, topf=(0,), **kwargs):
		super(SpectralPooling1D, self).__init__()
		if   "topf"  in kwargs:
			self.topf  = (int  (kwargs["topf" ][0]),)
			self.topf  = (self.topf[0]//2,)
		elif "gamma" in kwargs:
			self.gamma = (float(kwargs["gamma"][0]),)
			self.gamma = (self.gamma[0]/2,)
		else:
			raise RuntimeError("Must provide either topf= or gamma= !")
	def call(self, x, mask=None):
		xshape = x._keras_shape
		if hasattr(self, "topf"):
			topf = self.topf
		else:
			if KB.image_data_format() == "channels_first":
				topf = (int(self.gamma[0]*xshape[2]),)
			else:
				topf = (int(self.gamma[0]*xshape[1]),)
		
		if KB.image_data_format() == "channels_first":
			if topf[0] > 0 and xshape[2] >= 2*topf[0]:
				mask = [1]*(topf[0]              ) +\
					   [0]*(xshape[2] - 2*topf[0]) +\
					   [1]*(topf[0]              )
				mask = [[mask]]
				mask = np.asarray(mask, dtype=KB.floatx()).transpose((0,1,2))
				mask = KB.constant(mask)
				x   *= mask
		else:
			if topf[0] > 0 and xshape[1] >= 2*topf[0]:
				mask = [1]*(topf[0]              ) +\
					   [0]*(xshape[1] - 2*topf[0]) +\
					   [1]*(topf[0]              )
				mask = [[mask]]
				mask = np.asarray(mask, dtype=KB.floatx()).transpose((0,2,1))
				mask = KB.constant(mask)
				x   *= mask
		
		return x
class SpectralPooling2D(KL.Layer):
	def __init__(self, **kwargs):
		super(SpectralPooling2D, self).__init__()
		if   "topf"  in kwargs:
			self.topf  = (int  (kwargs["topf" ][0]), int  (kwargs["topf" ][1]))
			self.topf  = (self.topf[0]//2, self.topf[1]//2)
		elif "gamma" in kwargs:
			self.gamma = (float(kwargs["gamma"][0]), float(kwargs["gamma"][1]))
			self.gamma = (self.gamma[0]/2, self.gamma[1]/2)
		else:
			raise RuntimeError("Must provide either topf= or gamma= !")
	def call(self, x, mask=None):
		xshape = x._keras_shape
		if hasattr(self, "topf"):
			topf = self.topf
		else:
			if KB.image_data_format() == "channels_first":
				topf = (int(self.gamma[0]*xshape[2]), int(self.gamma[1]*xshape[3]))
			else:
				topf = (int(self.gamma[0]*xshape[1]), int(self.gamma[1]*xshape[2]))
		
		if KB.image_data_format() == "channels_first":
			if topf[0] > 0 and xshape[2] >= 2*topf[0]:
				mask = [1]*(topf[0]              ) +\
					   [0]*(xshape[2] - 2*topf[0]) +\
					   [1]*(topf[0]              )
				mask = [[[mask]]]
				mask = np.asarray(mask, dtype=KB.floatx()).transpose((0,1,3,2))
				mask = KB.constant(mask)
				x   *= mask
			if topf[1] > 0 and xshape[3] >= 2*topf[1]:
				mask = [1]*(topf[1]              ) +\
					   [0]*(xshape[3] - 2*topf[1]) +\
					   [1]*(topf[1]              )
				mask = [[[mask]]]
				mask = np.asarray(mask, dtype=KB.floatx()).transpose((0,1,2,3))
				mask = KB.constant(mask)
				x   *= mask
		else:
			if topf[0] > 0 and xshape[1] >= 2*topf[0]:
				mask = [1]*(topf[0]              ) +\
					   [0]*(xshape[1] - 2*topf[0]) +\
					   [1]*(topf[0]              )
				mask = [[[mask]]]
				mask = np.asarray(mask, dtype=KB.floatx()).transpose((0,3,1,2))
				mask = KB.constant(mask)
				x   *= mask
			if topf[1] > 0 and xshape[2] >= 2*topf[1]:
				mask = [1]*(topf[1]              ) +\
					   [0]*(xshape[2] - 2*topf[1]) +\
					   [1]*(topf[1]              )
				mask = [[[mask]]]
				mask = np.asarray(mask, dtype=KB.floatx()).transpose((0,1,3,2))
				mask = KB.constant(mask)
				x   *= mask
		
		return x

class MaxPoolingWithArgmax2D(KL.Layer):
    """
    This is only compatible with complex-valued inputs.
    The index of the max argument is according to which has the largest modulus within the kernel.
    """

    def __init__(self,
                 pool_size,
                 strides,
                 padding="SAME",
                 **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    def call(self, x, **kwargs):
        modulus = get_abs(x) # calculate modulus
        _, indices = tf.nn.max_pool_with_argmax(
            tf.concat([modulus, modulus], axis=-1),
            ksize=self.pool_size,
            strides=self.strides,
            padding=self.padding,
        )
        indices = tf.cast(indices, dtype=tf.int32) # turn int64 to int32
        tf_indices = tf.transpose(tf.unravel_index(tf.reshape(indices, [-1]), tf.shape(x)), [1,0]) # convert index which is support for gather_nd
        output = tf.reshape(tf.gather_nd(x, tf_indices), tf.shape(indices))
        return [output, indices]

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == "channels_first":
            rows = input_shape[2]
            cols = input_shape[3]
        else:
            rows = input_shape[1]
            cols = input_shape[2]
        rows = conv_utils.conv_output_length(
            rows, self.pool_size[0], self.padding, self.strides[0]
        )
        cols = conv_utils.conv_output_length(
            cols, self.pool_size[1], self.padding, self.strides[1]
        )
        if self.data_format == "channels_first":
            return tf.TensorShape([input_shape[0], input_shape[1], rows, cols])
        else:
            return tf.TensorShape([input_shape[0], rows, cols, input_shape[3]])

    def get_config(self):
        config = super().get_config()
        config.update({
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding
        })
        return config


class MaxUnpooling2D(KL.Layer):
    """
    SegNet Implementation for UpSampling layer
    """

    def __init__(self,
                 size,
                 **kwargs):
        super().__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, indices = inputs[0], inputs[1]
        indices = tf.cast(indices, dtype='int32')
        
        input_shape = tf.shape(updates, out_type='int32')
        batch_size = input_shape[0]
        if output_shape is None:
            output_shape = list(self.compute_output_shape(updates.shape))
            output_shape[0] = batch_size
        one_like_mask = tf.ones_like(indices, dtype='int32')
        batch_shape = KB.concatenate([[batch_size], [1], [1], [1]], axis=0)
        batch_range = tf.reshape(tf.range(batch_size, dtype='int32'), shape=batch_shape)
        b = one_like_mask * batch_range

        y = indices // (output_shape[2] * output_shape[3])
        x = (indices // output_shape[3]) % output_shape[2]
        feature_range = tf.range(output_shape[3], dtype='int32')
        f = one_like_mask * feature_range

        # transpose indices & reshape update values to one dimension
        updates_size = tf.size(updates)
        indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
        values = tf.reshape(updates, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)
        ret.set_shape(self.compute_output_shape(updates.shape))
        return ret

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        output_shape = [input_shape[0], input_shape[1] * self.size[0], input_shape[2] * self.size[1], input_shape[3]]
        return tuple(output_shape)

    def get_config(self):
        config = super(MaxUnpooling2D, self).get_config()
        config.update({
            'size': self.size,
        })
        return config

if __name__ == "__main__":
	import cv2, sys
	import __main__      as SP
	import fft_test      as CF
	
	# Build Model
	x = i = KL.Input(shape=(6,512,512))
	f = CF.FFT2()(x)
	p = SP.SpectralPooling2D(gamma=[0.15,0.15])(f)
	o = CF.IFFT2()(p)
	
	model = tf.keras.Model([i], [f,p,o])
	model.compile("sgd", "mse")
	
	# Use it
	img      = cv2.imread(sys.argv[1])
	imgBatch = img[np.newaxis,...].transpose((0,3,1,2))
	imgBatch = np.concatenate([imgBatch, np.zeros_like(imgBatch)], axis=1)
	f,p,o    = model.predict(imgBatch)
	ffted    = np.sqrt(np.sum(f[:,:3]**2 + f[:,3:]**2, axis=1))
	ffted    = ffted .transpose((1,2,0))/255
	pooled   = np.sqrt(np.sum(p[:,:3]**2 + p[:,3:]**2, axis=1))
	pooled   = pooled.transpose((1,2,0))/255
	filtered = np.clip(o,0,255).transpose((0,2,3,1))[0,:,:,:3].astype("uint8")
	
	# Display it
	cv2.imshow("Original", img)
	cv2.imshow("FFT",      ffted)
	cv2.imshow("Pooled",   pooled)
	cv2.imshow("Filtered", filtered)
	cv2.waitKey(0)
