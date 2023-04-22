# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:44:42 2021

@author: benzener
"""

import tensorflow as tf
from .utils_test import get_realpart, get_imagpart
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import get_custom_objects
import tensorflow.keras.backend as K




class AmplitudeMaxout(Layer):
    
    def __init__(self, num_pieces=4, name='AMU',**kwargs):
        super().__init__(**kwargs)
        self.__name__ = name
        self.num_pieces = num_pieces
        
    def call(self, x, axis=None):
        shape = x.get_shape().as_list() # [N,H,W,C]
        if axis is None:
            axis = -1
            shape[0] = -1
        if shape[axis]%2:
            raise ValueError(f'nb of real/imaginary channel are inequivalent')
        num_channels = shape[-1]//2 # C//2
        self.num_units = num_channels//self.num_pieces
        # padding to the multiple of num_pieces
        if num_channels%self.num_pieces:
            self.num_units += 1
            num_padding = self.num_pieces - num_channels%self.num_pieces
            padding_size = tf.concat([tf.shape(x)[:-1],tf.constant([num_padding])],axis=-1)
            zero_padding = tf.zeros(padding_size)
        # shape = [N,H,W,units]
        shape[axis] = self.num_units
        # expand shape = [N,H,W,units,pieces]
        exp_shape = shape + [self.num_pieces]
        real_part = x[:,:,:,:num_channels]
        imag_part = x[:,:,:,num_channels:]    
        if num_channels%self.num_pieces:
            real_part = tf.concat([real_part,zero_padding], axis=-1)
            imag_part = tf.concat([imag_part,zero_padding], axis=-1)
        real_part = tf.reshape(real_part, exp_shape)
        imag_part = tf.reshape(imag_part, exp_shape)
        real_part, imag_part = self.return_AMU(real_part, imag_part, exp_shape)
        return tf.concat([real_part,imag_part],axis=-1)    

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape[-1] = 2*self.num_units
        return tuple(shape)
    
    def get_config(self):
        base_config = super(AmplitudeMaxout, self).get_config()
        config = {'units': self.num_units,
                  'pieces': self.num_pieces}
        return dict(list(base_config.items()) + list(config.items()))
    
    def return_AMU(self, real_part, imag_part ,expand_shape):

        modulus = real_part**2 + imag_part**2
        expand_modulus = tf.reshape(modulus, expand_shape)    
        cond = tf.equal(expand_modulus,tf.reduce_max(expand_modulus,axis=-1,keepdims=True))
        real_part = tf.reduce_sum(real_part*tf.cast(cond,dtype=tf.float32),axis=-1)
        imag_part = tf.reduce_sum(imag_part*tf.cast(cond,dtype=tf.float32),axis=-1)
        return real_part, imag_part


class cReLU(Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        return tf.nn.relu(inputs)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
class cLeakyReLU(Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        return tf.nn.leaky_relu(inputs)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
class ctanh(Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        return tf.nn.tanh(inputs)
    
    def compute_output_shape(self, input_shape):
        return input_shape

class zReLU(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        real = get_realpart(inputs)
        imag = get_imagpart(inputs)
        real_t = tf.transpose(real, [0,3,1,2])
        imag_t = tf.transpose(imag, [0,3,1,2])
        cond = (real_t >= 0) & (imag_t >= 0)
        real_t = tf.where(cond, real_t, 0.)
        imag_t = tf.where(cond, imag_t, 0.)
        real = tf.transpose(real_t, [0,2,3,1])
        imag = tf.transpose(imag_t, [0,2,3,1])
        return tf.cast(tf.concat([real,imag], axis=-1),dtype=tf.float32)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
# from tensorflow.keras.layers import Layer
class FLeakyReLU(Layer):
    def __init__(self, alpha=0.01, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
    

    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        self.threshold = self.add_weight(shape=param_shape,
                                          name='threshold',
                                          initializer='zeros')
        self.built = True
        
    def call(self, inputs):
        '''
            FLeakyReLU(x) = alpha*(x+|threshold|), if x < -|threshold|
                          = x - |threshold|, if |threshold| < x
                          = 0, otherwise
        '''
        # For any positive value. Truncated to zero if it's less than the threshold.
        positive = tf.nn.relu(inputs - tf.abs(self.threshold))
        # For any negative value. Truncated to zero if it's larger than the threshold.
        negative = self.alpha * -tf.nn.relu(-inputs - tf.abs(self.threshold))
        # finally, combine the two regions
        return positive + negative

    def get_config(self):
        # any declared variables inside __init__ are required to be in config
        base_config = super().get_config()
        config = {'alpha': self.alpha}
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
# class FLeakyReLU(Layer):
#     def __init__(self, threshold=1e-2, alpha=0.01, **kwargs):
#         super().__init__(**kwargs)
#         self.threshold = threshold
#         self.alpha = alpha
        
#     def call(self, inputs):
#         less_than_threshold = tf.multiply(
#             tf.cast(tf.math.less_equal(inputs, -self.threshold),
#                               dtype=tf.float32),self.alpha)
#         greater_than_threshold = tf.cast(tf.math.greater_equal(inputs, self.threshold),  dtype=tf.float32)
#         return tf.multiply(tf.add(less_than_threshold,greater_than_threshold),inputs)
    
 
#     def get_config(self):
#         base_config = super().get_config()
#         config = {'threshold': self.threshold,
#                   'alpha': self.alpha}
#         return dict(list(base_config.items()) + list(config.items()))

#     def compute_output_shape(self, input_shape):
#         return input_shape

'''
ValueError: including someone isn't differentiable, especially
 
x = tf.constant(tf.random.normal((10,),dtype=tf.float32))
y = tf.constant(np.arange(10),dtype=tf.float32)
b = tf.Variable(tf.ones(10),dtype=tf.float32)
with tf.GradientTape(persistent=True) as g:
    g.watch(b)
    cond = tf.greater_equal(tf.add(z, b),0)
    cond = tf.cast(cond, tf.float32)
g.gradient(cond, b) is None, since cond is a boolean type w.r.t b that can't derive gradient


'''

# class modReLU(Layer):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
        
#     def build(self, input_shape):
#         b_init = tf.zeros_initializer()
#         self.b = tf.Variable(
#                 initial_value=lambda: b_init(shape=(input_shape[-1]//2),dtype='float32'),
#                 trainable=True,
#                 name='b')
#         super().build(input_shape)
        
#     def call(self, inputs):
#         real = get_realpart(inputs)
#         imag = get_imagpart(inputs)
#         modulus = tf.sqrt(tf.add(tf.pow(real,2),tf.pow(imag,2)))
#         phase = tf.atan2(imag, real)
#         cond = tf.greater_equal(tf.add(modulus, self.b),0)
#         modulus = tf.where(cond, modulus, 0)
#         real = modulus*tf.cos(phase)
#         imag = modulus*tf.sin(phase)
#         return tf.concat([real, imag], axis=-1)

#     def get_config(self):
#         base_config = super().get_config()
#         config = {'b': self.b}
#         return dict(list(base_config.items()) + list(config.items()))
    
#     def compute_output_shape(self, input_shape):
#         return input_shape


class modReLU(Layer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.b_factor = None
        
    def build(self, input_shape):
        self.b_factor = self.add_weight(shape=(input_shape[-1]//2,),
                                          name='b_factor',
                                          initializer='zeros',
                                          trainable=True)
        
    def call(self, inputs):
        real = get_realpart(inputs)
        imag = get_imagpart(inputs)
        modulus = tf.sqrt(tf.add(tf.pow(real,2),tf.pow(imag,2)))
        '''
        ReLU(|z|+b)exp(i*theta) = ReLU(|z|+b)(cos(theta)+isin(theta))
        cos(theta) = Real_part/modulus, sin(theta) = Imag_part/modulus
        '''
        real = tf.multiply(tf.math.divide_no_nan(real,modulus),tf.nn.relu(tf.add(modulus,self.b_factor)))
        imag = tf.multiply(tf.math.divide_no_nan(imag,modulus),tf.nn.relu(tf.add(modulus,self.b_factor)))
        return tf.concat([real, imag], axis=-1)
 
    def get_config(self):
        base_config = super().get_config()
        config = {'b_factor': self.b_factor}
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
        return input_shape    
#        

'''
    add custom functions to keras alias
    
    usage:
        complexnn.conv.ComplexConv2D(filters, size, activation='zReLU')
    
    otherwise, advanced activation function should be used like any other layer independently:
        model.add(complexnn.activation.cReLU())
        outputs = complexnn.activation.cReLU()(inputs)
        
'''
get_custom_objects().update({'zReLU': zReLU(),
                             'cReLU': cReLU(),
                             'cLeakyReLU': cLeakyReLU(),
                             'ctanh': ctanh(),
                             'modReLU': modReLU(),
                             'FLeakyReLU': FLeakyReLU()})

