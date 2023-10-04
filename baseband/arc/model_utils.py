# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 12:37:48 2023

@author: benzener
"""
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
import tensorflow as tf
from tensorflow.keras.layers import Layer
from complexnn.utils_test import get_realpart, get_imagpart, get_abs
import tensorflow.keras.backend as K
import math


class InOne(tf.keras.constraints.Constraint):
    def __init__(self, min_value=0.0, max_value=1.0):
        self.min_value = min_value
        self.max_value = max_value
    def __call__(self, w):
        # w = w * tf.cast(tf.math.less_equal(w, 1.), w.dtype)
        return tf.clip_by_value(w, self.min_value, self.max_value)
    def get_config(self):
        return {
            "min_value": self.min_value,
            "max_value": self.max_value,
        }
     
class CosineDecay(LearningRateSchedule):
    """A LearningRateSchedule that uses a cosine decay schedule.

    See [Loshchilov & Hutter, ICLR2016](https://arxiv.org/abs/1608.03983),
    SGDR: Stochastic Gradient Descent with Warm Restarts.

    When training a model, it is often useful to lower the learning rate as
    the training progresses. This schedule applies a cosine decay function
    to an optimizer step, given a provided initial learning rate.
    It requires a `step` value to compute the decayed learning rate. You can
    just pass a TensorFlow variable that you increment at each training step.

    The schedule is a 1-arg callable that produces a decayed learning
    rate when passed the current optimizer step. This can be useful for changing
    the learning rate value across different invocations of optimizer functions.
    It is computed as:

    ```python
    def decayed_learning_rate(step):
      step = min(step, decay_steps)
      cosine_decay = 0.5 * (1 + cos(pi * step / decay_steps))
      decayed = (1 - alpha) * cosine_decay + alpha
      return initial_learning_rate * decayed
    ```

    Example usage:
    ```python
    decay_steps = 1000
    lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps)
    ```

    You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
    as the learning rate. The learning rate schedule is also serializable and
    deserializable using `tf.keras.optimizers.schedules.serialize` and
    `tf.keras.optimizers.schedules.deserialize`.

    Returns:
      A 1-arg callable learning rate schedule that takes the current optimizer
      step and outputs the decayed learning rate, a scalar `Tensor` of the same
      type as `initial_learning_rate`.
    """

    def __init__(
        self, initial_learning_rate, decay_steps, alpha=0.0, name=None
    ):
        """Applies cosine decay to the learning rate.

        Args:
          initial_learning_rate: A scalar `float32` or `float64` Tensor or a
            Python number. The initial learning rate.
          decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
            Number of steps to decay over.
          alpha: A scalar `float32` or `float64` Tensor or a Python number.
            Minimum learning rate value as a fraction of initial_learning_rate.
          name: String. Optional name of the operation.  Defaults to
            'CosineDecay'.
        """
        super().__init__()

        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "CosineDecay"):
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate"
            )
            dtype = initial_learning_rate.dtype
            decay_steps = tf.cast(self.decay_steps, dtype)

            global_step_recomp = tf.cast(step, dtype)
            global_step_recomp = tf.minimum(global_step_recomp, decay_steps)
            completed_fraction = global_step_recomp / decay_steps
            cosine_decayed = 0.5 * (
                1.0
                + tf.cos(tf.constant(math.pi, dtype=dtype) * completed_fraction)
            )

            decayed = (1 - self.alpha) * cosine_decayed + self.alpha
            return tf.multiply(initial_learning_rate, decayed)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha,
            "name": self.name,
        }
    
class ComplexNormalization(Layer):
    '''
    Complex-valued normalization in tensorflow.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        

    def call(self, inputs):
        real = get_realpart(inputs)
        imag = get_imagpart(inputs)
        modulus = get_abs(inputs)
        ratio = tf.reduce_max(modulus, axis=(1,2,3), keepdims=True)
        real = real/ (ratio + K.epsilon())
        imag = imag/ (ratio + K.epsilon())       
        return tf.concat([real, imag], axis=-1)

    def get_config(self):
        base_config = super().get_config()
        return base_config

    def compute_output_shape(self, input_shape):
        return input_shape

class ComplexConcatenate(Layer):
    '''
    '''
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
    
    def call(self, tensors):
        real_buffer = []
        imag_buffer = []
        for x in tensors:
            real_buffer.append(get_realpart(x))
            imag_buffer.append(get_imagpart(x))
        real = tf.concat(real_buffer, self.axis)
        imag = tf.concat(imag_buffer, self.axis)
        return tf.concat([real,imag], axis=-1)

    
    def compute_output_shape(self, input_shape):
        if (not isinstance(input_shape, (tuple, list))) or (not isinstance(input_shape[0], (tuple, list))):
            # The tf_utils.shape_type_conversion decorator turns tensorshapes
            # into tuples, so we need to verify that `input_shape` is a
            # list/tuple, *and* that the individual elements are themselves
            # shape tuples.
            raise ValueError(
                "A `Concatenate` layer should be called on a list of inputs. "
                f"Received: input_shape={input_shape}"
            )
        input_shapes = input_shape
        output_shape = list(input_shapes[0])
        for shape in input_shapes[1:]:
            if output_shape[self.axis] is None or shape[self.axis] is None:
                output_shape[self.axis] = None
                break
            output_shape[self.axis] += shape[self.axis]
        return tuple(output_shape)
        
    def get_config(self):
        config = {
            "axis": self.axis,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class GlobalAveragePooling(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, tensors):
        avg = tf.math.reduce_mean(tensors, axis=(1,2))
        return avg
    
    def compute_output_shape(self, input_shape):
        return(input_shape[0], input_shape[-1])
    

        
