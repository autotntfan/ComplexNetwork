# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 00:26:54 2023

@author: benzener
"""

from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import tensorflow as tf
from .utils_test import get_realpart, get_imagpart

class PixelShuffler(Layer):
    # reference:https://github.com/fengwang/subpixel_conv2d/blob/master/subpixel_conv2d.py
    def __init__(self, upsampling_factor=(2,2), **kwargs):
        super().__init__(**kwargs)
        self.upsampling_factor = upsampling_factor

    def build(self, input_shape):
        last_dim = input_shape[-1]
        factor = self.upsampling_factor[0] * self.upsampling_factor[1]
        if last_dim % (factor) != 0:
            raise ValueError('Channel ' + str(last_dim) + ' should be of '
                             'integer times of upsampling_factor^2: ' +
                             str(factor) + '.')

    def call(self, inputs):
        real = get_realpart(inputs)
        imag = get_imagpart(inputs)
        real_shuffle = tf.nn.depth_to_space(real, self.upsampling_factor[0])
        imag_shuffle = tf.nn.depth_to_space(imag, self.upsampling_factor[0])
        return tf.concat([real_shuffle, imag_shuffle], axis=-1)

    def get_config(self):
        config = { 'upsampling_factor': self.upsampling_factor, }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        factor = self.upsampling_factor * self.upsampling_factor
        input_shape_1 = None
        if input_shape[1] is not None:
            input_shape_1 = input_shape[1] * self.upsampling_factor
        input_shape_2 = None
        if input_shape[2] is not None:
            input_shape_2 = input_shape[2] * self.upsampling_factor
        dims = [ input_shape[0],
                 input_shape_1,
                 input_shape_2,
                 int(input_shape[3]/factor)
               ]
        return tuple(dims)

    
# class PixelShuffler(Layer):
#     # reference:https://gist.github.com/t-ae/6e1016cc188104d123676ccef3264981
#     def __init__(self, size=(2, 2), **kwargs):
#         super(PixelShuffler, self).__init__(**kwargs)
#         self.size = conv_utils.normalize_tuple(size, 2, 'size')

#     def call(self, inputs):
#         input_shape = inputs.shape
#         if len(input_shape) != 4:
#             raise ValueError('Inputs should have rank ' +
#                              str(4) +
#                              '; Received input shape:', str(input_shape))
#         real = get_realpart(inputs)
#         imag = get_imagpart(inputs)
#         batch_size, h, w, c = real.shape
#         if batch_size is None:
#             batch_size = -1
#         rh, rw = self.size
#         oh, ow = h * rh, w * rw
#         oc = c // (rh * rw)

#         real_out = K.reshape(real, (batch_size, h, w, rh, rw, oc))
#         real_out = K.permute_dimensions(real_out, (0, 1, 3, 2, 4, 5))
#         real_out = K.reshape(real_out, (batch_size, oh, ow, oc))
#         imag_out = K.reshape(imag, (batch_size, h, w, rh, rw, oc))
#         imag_out = K.permute_dimensions(imag_out, (0, 1, 3, 2, 4, 5))
#         imag_out = K.reshape(imag_out, (batch_size, oh, ow, oc))
#         return tf.concat([real_out,imag_out], axis=-1)

#     def compute_output_shape(self, input_shape):
#         if len(input_shape) != 4:
#             raise ValueError('Inputs should have rank ' +
#                              str(4) +
#                              '; Received input shape:', str(input_shape))
#         height = input_shape[1] * self.size[0] if input_shape[1] is not None else None
#         width = input_shape[2] * self.size[1] if input_shape[2] is not None else None
#         channels = input_shape[3] // self.size[0] // self.size[1]

#         if channels * self.size[0] * self.size[1] != input_shape[3]:
#             raise ValueError('channels of input and size are incompatible')

#         return (input_shape[0],
#                 height,
#                 width,
#                 channels)

#     def get_config(self):
#         config = {'size': self.size}
#         base_config = super(PixelShuffler, self).get_config()

#         return dict(list(base_config.items()) + list(config.items()))
'''   
class PixelShuffler(Layer):
    # Real-valued Pixel Shuffle
    # reference:https://gist.github.com/t-ae/6e1016cc188104d123676ccef3264981
    def __init__(self, size=(2, 2), data_format=None, **kwargs):
        super(PixelShuffler, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')

    def call(self, inputs):
        input_shape = inputs.shape
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' +
                             str(4) +
                             '; Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            batch_size, c, h, w = input_shape
        elif self.data_format == 'channels_last':
            batch_size, h, w, c = input_shape
        if batch_size is None:
            batch_size = -1
        rh, rw = self.size
        oh, ow = h * rh, w * rw
        oc = c // (rh * rw)
        if self.data_format == 'channels_first':
            out = tf.reshape(inputs, (batch_size, rh, rw, oc, h, w))
            out = tf.transpose(out, (0, 3, 4, 1, 5, 2))
            out = tf.reshape(out, (batch_size, oc, oh, ow))
        elif self.data_format == 'channels_last':
            out = K.reshape(inputs, (batch_size, h, w, rh, rw, oc))
            out = K.permute_dimensions(out, (0, 1, 3, 2, 4, 5))
            out = K.reshape(out, (batch_size, oh, ow, oc))
        return out

    def compute_output_shape(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' +
                             str(4) +
                             '; Received input shape:', str(input_shape))

        if self.data_format == 'channels_first':
            height = input_shape[2] * self.size[0] if input_shape[2] is not None else None
            width = input_shape[3] * self.size[1] if input_shape[3] is not None else None
            channels = input_shape[1] // self.size[0] // self.size[1]

            if channels * self.size[0] * self.size[1] != input_shape[1]:
                raise ValueError('channels of input and size are incompatible')

            return (input_shape[0],
                    channels,
                    height,
                    width)

        elif self.data_format == 'channels_last':
            height = input_shape[1] * self.size[0] if input_shape[1] is not None else None
            width = input_shape[2] * self.size[1] if input_shape[2] is not None else None
            channels = input_shape[3] // self.size[0] // self.size[1]

            if channels * self.size[0] * self.size[1] != input_shape[3]:
                raise ValueError('channels of input and size are incompatible')

            return (input_shape[0],
                    height,
                    width,
                    channels)

    def get_config(self):
        config = {'size': self.size,
                  'data_format': self.data_format}
        base_config = super(PixelShuffler, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
'''