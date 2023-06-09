# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 17:19:21 2023

@author: benzener
"""
import tensorflow as tf
import os
import numpy as np
import time
from complexnn.activation import cReLU, zReLU, modReLU, AmplitudeMaxout, FLeakyReLU, ctanh
from complexnn.loss import ComplexRMS, ComplexMAE, ComplexMSE, SSIM, MS_SSIM, SSIM_MSE, aSSIM_MSE, ssim_map
from complexnn.conv_test import ComplexConv2D
from complexnn.bn_test import ComplexBatchNormalization
from complexnn.pool_test import MaxPoolingWithArgmax2D, MaxUnpooling2D
from complexnn.pixelshuffle import PixelShuffler
from tensorflow.keras.layers import Input, Add, Conv2D, Conv2DTranspose, BatchNormalization, Dropout, Concatenate, LeakyReLU, MaxPool2D
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.activations import tanh
from datetime import datetime
if __name__ == '__main__':
    import sys
    currentpath = os.getcwd()
    addpath = os.path.dirname(os.path.dirname(currentpath))
    if addpath not in sys.path:
        sys.path.append(addpath)
    from baseband.setting import constant
    from baseband.utils.fig_utils import envelope_fig
    sys.path.remove(addpath)
else:
    from ..setting import constant
    from ..utils.fig_utils import envelope_fig


class BasedModel(): 
    def __init__(self,
                 filters=4,
                 size=(3,3),
                 batch_size=2,
                 lr=1e-4,
                 epochs=8,
                 validation_split=0,
                 validation_data=None,
                 seed=7414,
                 activations='LeakyReLU',
                 losses='ComplexMSE',
                 forward=False,
                 dropout_rate=None,
                 callbacks=None,
                 use_bias=False,
                 complex_network=True,
                 num_downsample_layer=5,
                 batchsizeschedule=False):
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
        self.dropout_rate = dropout_rate
        self.callbacks = callbacks
        self.use_bias = use_bias
        self.complex_network = complex_network
        self.forward = forward
        self.num_downsample_layer = num_downsample_layer
        self.batchsizeschedule = batchsizeschedule

        self.convFunc = ComplexConv2D if self.complex_network else Conv2D
        self.bnFunc = ComplexBatchNormalization if self.complex_network else BatchNormalization
        self.fine_tune = False
           
    def downsample_block_conv(self, x, filters, kernel_size, dropout_rate=None):
        x = self.convFunc(filters, kernel_size, padding='same', use_bias=self.use_bias)(x)
        x = self.bnFunc()(x)
        x = self.activations()(x)
        x = self.convFunc(filters, kernel_size, strides=2, padding='same', use_bias=self.use_bias)(x)
        x = self.bnFunc()(x)
        x = self.activations()(x)
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
        return x
    
    def downsample_block_maxpool(self, x, filters, kernel_size, dropout_rate=None):
        x = self.convFunc(filters, kernel_size, padding='same', use_bias=self.use_bias)(x)
        x = self.bnFunc()(x)
        x = self.activations()(x)
        x, ind = MaxPoolingWithArgmax2D((2,2),2)(x)
        x = self.bnFunc()(x)
        x = self.activations()(x)
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
        return x, ind
    
    def upsample_block_PS(self, x, filters, kernel_size, dropout_rate=None):
        # pixel shuffle
        x = PixelShuffler((2,2))(x)
        x = self.bnFunc()(x)
        x = self.activations()(x)
        x = self.convFunc(filters, kernel_size, padding='same', use_bias=self.use_bias)(x)
        x = self.bnFunc()(x)
        x = self.activations()(x)
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
        return x

    def upsample_block_transp(self, x, filters, kernel_size, dropout_rate=None):
        if self.complex_network:
            x = self.convFunc(filters, kernel_size, strides=2, padding='same', transposed=True)(x)
        else:
            x = Conv2DTranspose(filters, kernel_size, strides=2, padding='same')(x)
        x = self.bnFunc()(x)
        x = self.activations()(x)
        x = self.convFunc(filters, kernel_size, padding='same', use_bias=self.use_bias)(x)
        x = self.bnFunc()(x)
        x = self.activations()(x)
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
        return x
    
    def upsample_block_unpool(self, x, filters, kernel_size, dropout_rate=None):
        x = MaxUnpooling2D((2,2))(x)
        x = self.bnFunc()(x)
        x = self.activations()(x)
        x = self.convFunc(filters, kernel_size, padding='same', use_bias=self.use_bias)(x)
        x = self.bnFunc()(x)
        x = self.activations()(x)
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
        return x
    
    def identity_block(self, x, filters, kernel_size, dropout_rate=None):
        last_layer_filters = x.shape[-1]//2 if self.complex_network else x.shape[-1]
        shortcut = x
        x = self.convFunc(filters, (1,1), padding='same', strides=1, use_bias=self.use_bias)(x)
        x = self.bnFunc()(x)
        x = self.activations()(x)
        x = self.convFunc(filters, kernel_size, padding='same', strides=1, use_bias=self.use_bias)(x)
        x = self.bnFunc()(x)
        x = self.activations()(x)
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
        x = self.convFunc(last_layer_filters, (1,1), padding='same', strides=1, use_bias=self.use_bias)(x)
        x = self.bnFunc()(x)
        x = self.activations()(x)
        x = Add()([x, shortcut])
        x = self.activations()(x)
        return x
    
    def conv_block(self, x, filters, kernel_size, stride, dropout_rate=None):
        last_layer_filters = x.shape[-1]//2 if self.complex_network else x.shape[-1]
        shortcut = x
        x = self.convFunc(filters, (1,1), padding='same', strides=stride, use_bias=self.use_bias)(x)
        x = self.bnFunc()(x)
        x = self.activations()(x)
        x = self.convFunc(filters, kernel_size, padding='same', strides=1, use_bias=self.use_bias)(x)
        x = self.bnFunc()(x)
        x = self.activations()(x)
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
        x = self.convFunc(last_layer_filters, (1,1), padding='same', strides=1, use_bias=self.use_bias)(x)
        x = self.bnFunc()(x)
        x = self.activations()(x)
        shortcut = self.convFunc(last_layer_filters, (1,1), strides=stride, padding='same', use_bias=self.use_bias)(shortcut)
        shortcut = self.bnFunc()(shortcut)
        x = Add()([x, shortcut])
        x = self.activations()(x)
        return x
    


        
    def _sanitized(self):
        if self.complex_network:
            assert self.input_shape[-1] == 2
            if isinstance(self.losses, str):
                if self.losses not in {'ComplexRMS', 'ComplexMAE', 'ComplexMSE','SSIM','MS_SSIM','SSIM_MSE', 'aSSIM_MSE', 'SSIM_map'}:
                    raise KeyError('Invalid complex-valued loss function')
            if isinstance(self.activations, str):
                if self.activations not in {'cReLU', 'zReLU', 'modReLU', 'LeakyReLU', 'AMU', 'FLeakyReLU'}:
                    raise  KeyError('Unsupported activation')
        else:
            assert self.input_shape[-1] == 1
            if isinstance(self.losses, str):
                if self.losses not in {'MSE','SSIM', 'MS_SSIM', 'SSIM_MSE', 'aSSIM_MSE', 'SSIM_map'}:
                    raise KeyError('Invalid real-valued loss function')
            if isinstance(self.activations, str):
                if self.activations not in {'LeakyReLU','FLeakyReLU'}:
                    raise  KeyError('Unsupported activation')
    def get_custom_object(self):
        custom_object = {
            'ComplexConv2D':ComplexConv2D,
            'ComplexBatchNormalization':ComplexBatchNormalization,
            'MaxPoolingWithArgmax2D':MaxPoolingWithArgmax2D,
            'MaxUnpooling2D':MaxUnpooling2D,
            'ComplexMSE':ComplexMSE,
            'ctanh':ctanh,
            'FLeakyReLU': FLeakyReLU
            }
        return custom_object
    
    def _generate_name(self):
        now = datetime.now()
        day_month_year = now.strftime("%d%m%Y")
        data_type = 'complex' if self.complex_network else 'real'
        forward = 'forward' if self.forward else 'Notforward'
        model_name = self.__class__.__name__
        epochs = str(self.epochs)
        if self.fine_tune:
            self.model_name = f'{data_type}{model_name}_{forward}_{epochs}_{self.losses}_{self.activations}_finetune' + day_month_year
        else:
            self.model_name = f'{data_type}{model_name}_{forward}_{epochs}_{self.losses}_{self.activations}_' + day_month_year
    
    def _build_info(self):
        self.model_info = {
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
            'dropout_rate':self.dropout_rate,
            'use_bias':self.use_bias,
            'batchsizeschedule':self.batchsizeschedule
            }
        if self.validation_data is not None:
            cache_file = os.path.join(constant.CACHEPATH, 'parameters.txt')
            with open(cache_file, 'r') as f:
                content = f.read() # type of content is string
            cache_info = eval(content) # convert string to dict
            self.model_info['validation_split'] = cache_info['validation_split']

    def _convert_act(self):
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
    
    def _convert_loss(self):
        # determine loss function 
        if self.losses in {'ComplexRMS', 'ComplexMAE', 'ComplexMSE', 'SSIM', 'MSE','MS_SSIM' ,'SSIM_MSE', 'aSSIM_MSE', 'SSIM_map'}:
            self.losses = {
                'ComplexRMS': ComplexRMS,
                'ComplexMAE': ComplexMAE,
                'ComplexMSE': ComplexMSE,
                'SSIM'      : SSIM,
                'MS_SSIM'   : MS_SSIM,
                'SSIM_MSE'  : SSIM_MSE,
                'MSE'       : MeanSquaredError(),
                'aSSIM_MSE' : aSSIM_MSE,
                'SSIM_map': ssim_map
                }[self.losses]
                      
       
    def training(self, x, y, model_name=None):
        '''
            without supporting Forward model
        '''
        
        history = {
            'loss':[]
            }
        x_train, y_train = x, y
        
        if self.validation_data is not None:
            x_val, y_val = self.validation_data
        elif self.validation_rate:
            num_train = round(x.shape[0]*(1-self.validation_rate))
            x_train, y_train = x[:num_train], y[:num_train]
            x_val, y_val = x[num_train:], y[num_train:]
        N = x_train.shape[0]
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(N).batch(self.batch_size)
        self.input_shape = x_train.shape[1:]
        self._sanitized()
        self._build_info()

        if model_name is None:
            self._generate_name()
            self._convert_loss()
            self._convert_act()
            self.model = self.build_model()
        else:
            self.model_info['basedmodel'] = model_name
            self.model = tf.keras.models.load_model(os.path.join(constant.MODELPATH,model_name,model_name+'.h5'),custom_objects=self.custom_object)
            self.fine_tune = True
            self._generate_name()

        try:
            N = x_val.shape[0]
            valid_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).shuffle(N).batch(self.batch_size)
            no_validation = False
            history['val_loss'] = []
            envelope_fig(y_val[5:6],60, 0, 
                         "Ground truth validation", 
                         saved_name="Ground_truth_val",
                         model_name=self.model_name,
                         saved_dir='PredictionEachEpoch')
        except Exception:
            no_validation = True
        envelope_fig(y_train[4:5],60, 0, 
                     "Ground truth", 
                     saved_name="Ground_truth",
                     model_name=self.model_name,
                     saved_dir='PredictionEachEpoch')
        
        
        self.model.summary()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        
        for epoch in range(self.epochs):
            try:
                self.losses = self.losses(epoch/self.epochs)
            except TypeError:
                pass
            if self.batchsizeschedule and epoch != 0:
                batch_size = int(1 + self.batch_size//self.epochs)
                train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(N).batch(batch_size)
            s = time.time()
            loss_train_epoch = []
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                loss_train_epoch.append(self.train_step(x_batch_train, y_batch_train))
            history['loss'].append(np.mean(loss_train_epoch))
            if no_validation:
                e = time.time()
                print(f'Epoch:{epoch+1}/{self.epochs} - {(e-s):.1f}s - ' \
                      f'loss:{np.mean(loss_train_epoch):.6f} - \n')
            else:
                loss_valid_epoch = []
                for x_batch_val, y_batch_val in valid_dataset:
                    loss_valid_epoch.append(self.test_step(x_batch_val, y_batch_val))
                history['val_loss'].append(np.mean(loss_valid_epoch))
                e = time.time()
                print(f'Epoch:{epoch+1}/{self.epochs} - {(e-s):.1f}s - ' \
                      f'loss:{np.mean(loss_train_epoch):.6f} - ' \
                      f'val_loss:{np.mean(loss_valid_epoch):.6f} \n')
            if (epoch+1)%10 == 0:
                train_img = self.model.predict(x_train[:self.batch_size])
                envelope_fig(train_img[4:5],60, 0, 
                             f"Prediction-epoch-{epoch+1}", 
                             saved_name=f"epoch-{epoch+1}",
                             model_name=self.model_name,
                             saved_dir='PredictionEachEpoch')
                if not no_validation:
                    val_img = self.model.predict(x_val[:self.batch_size])
                    envelope_fig(val_img[3:4],60, 0, 
                                 f"Validation-epoch-{epoch+1}", 
                                 saved_name=f"epoch-val-{epoch+1}",
                                 model_name=self.model_name,
                                 saved_dir='PredictionEachEpoch')
            
            # if (epoch+1)%50 == 0:
            #     print('saving')
            #     save_path = os.path.join(constant.MODELPATH, model_name, f'checkpoints/chkpt-epoch{epoch+1:04d}/')
            #     if os.path.exists(save_path):
            #         os.makedirs(save_path)
            #     self.model.save_weights(save_path)
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
    
class UNet(BasedModel):
    def __init__(self,
                 filters=8,
                 size=(3,3),
                 batch_size=8,
                 lr=tf.keras.optimizers.schedules.PiecewiseConstantDecay([200//3,2*200//3],[1e-3,5e-4,1e-4]),
                 epochs=200,
                 seed=7414,
                 activations='FLeakyReLU',
                 losses='SSIM',
                 forward=False,
                 dropout_rate=None,
                 use_bias=False,
                 complex_network=True,
                 num_downsample_layer=5,
                 **kwargs):
        super().__init__(
            filters=filters,
            size=size,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            seed=seed,
            activations=activations,
            losses=losses,
            forward=forward,
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            complex_network=complex_network,
            num_downsample_layer=num_downsample_layer,
            **kwargs)
        
    def build_model(self):
        tf.random.set_seed(self.seed)
        inputs = Input(self.input_shape)
        if self.forward:
            inputs_forward = Input(self.input_shape)            
        x = inputs
        # downsample
        skips = []
        for ithlayer in range(1,self.num_downsample_layer+1):
            x = self.downsample_block_conv(x, 2**(ithlayer+1)*self.filters, self.size) # downsampling 2^num_downsample_layer
            skips.append(x)
        x = self.convFunc(2**(ithlayer+1)*self.filters, self.size, padding='same', use_bias=self.use_bias)(x)
        x = self.bnFunc()(x)
        x = self.activations()(x)
        # upsample
        skips = reversed(skips)
        for skip in skips:
            x = Concatenate()([x, skip])
            x = self.upsample_block_transp(x, 2**ithlayer*self.filters, self.size)
            ithlayer = ithlayer - 1
        for _ in range(3):
            x = self.convFunc(2*self.filters, 3, padding='same', use_bias=self.use_bias)(x)
            x = self.bnFunc()(x)
            x = self.activations()(x)
        x = self.convFunc(1, 3, padding='same', use_bias=self.use_bias)(x)
        x = self.bnFunc()(x)
        if self.forward:
            x = self.activations()(x)
            x = Concatenate(axis=-1)((x, inputs_forward))
            x = self.convFunc(2, 3, padding='same', use_bias=self.use_bias)(x)
            x = self.bnFunc()(x)
            x = tf.keras.layers.LeakyReLU()(x)
            return tf.keras.Model(inputs=[inputs, inputs_forward], outputs=x)
        else:    
            x = tf.keras.layers.LeakyReLU()(x)
            return tf.keras.Model(inputs=inputs, outputs=x) 

class SegNet(BasedModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build_model(self):
        tf.random.set_seed(self.seed)
        inputs = Input(self.input_shape)
        if self.forward:
            inputs_forward = Input(self.input_shape)            
        x = inputs
        # downsample
        inds = []
        for ithlayer in range(1,self.num_downsample_layer+1):
            x, ind = self.downsample_block_maxpool(x, 2**(ithlayer+1)*self.filters, self.size) # downsampling 2^num_downsample_layer
            inds.append(ind)
        x = self.convFunc(2**(ithlayer+1)*self.filters, self.size, padding='same', use_bias=self.use_bias)(x)
        x = self.bnFunc()(x)
        x = self.activations()(x)
        # upsample
        inds = reversed(inds)
        for ind in inds:
            x = self.upsample_block_unpool([x,ind], 2**ithlayer*self.filters, self.size)
            ithlayer = ithlayer - 1            
        for _ in range(3):
            x = self.convFunc(2*self.filters, 3, padding='same', use_bias=self.use_bias)(x)
            x = self.bnFunc()(x)
            x = self.activations()(x)  
        x = self.convFunc(1, (1,1), padding='same', use_bias=self.use_bias)(x)
        x = self.bnFunc()(x)
        if self.forward:
            x = self.activations()(x)
            x = Concatenate(axis=-1)((x, inputs_forward))
            x = self.convFunc(2, 3, padding='same', use_bias=self.use_bias)(x)
            x = self.bnFunc()(x)
            x = tf.keras.layers.LeakyReLU()(x)
            return tf.keras.Model(inputs=[inputs, inputs_forward], outputs=x)
        else:    
            x = tf.keras.layers.LeakyReLU()(x)
            return tf.keras.Model(inputs=inputs, outputs=x)
        
class PSNet(BasedModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build_model(self):
        tf.random.set_seed(self.seed)
        inputs = Input(self.input_shape)
        if self.forward:
            inputs_forward = Input(self.input_shape)            
        x = inputs
        # downsample
        inds = []
        for ithlayer in range(1,self.num_downsample_layer+1):
            x, ind = self.downsample_block_maxpool(x, 4**(ithlayer+1)*self.filters, self.size) # downsampling 2^num_downsample_layer
            inds.append(ind)
        x = self.convFunc(2**(ithlayer+1)*self.filters, self.size, padding='same', use_bias=self.use_bias)(x)
        x = self.bnFunc()(x)
        x = self.activations()(x)
        # upsample
        inds = reversed(inds)
        for ind in inds:
            x = self.upsample_block_PS(x, 4**ithlayer*self.filters, self.size)
            ithlayer = ithlayer - 1      
        for _ in range(3):
            x = self.convFunc(2*self.filters, 3, padding='same', use_bias=self.use_bias)(x)
            x = self.bnFunc()(x)
            x = self.activations()(x)  
        x = self.convFunc(1, 3, padding='same', use_bias=self.use_bias)(x)
        x = self.bnFunc()(x)
        if self.forward:
            x = self.activations()(x)
            x = Concatenate(axis=-1)((x, inputs_forward))
            x = self.convFunc(2, 3, padding='same', use_bias=self.use_bias)(x)
            x = self.bnFunc()(x)
            x = tf.keras.layers.LeakyReLU()(x)
            return tf.keras.Model(inputs=[inputs, inputs_forward], outputs=x)
        else:    
            x = tf.keras.layers.LeakyReLU()(x)
            return tf.keras.Model(inputs=inputs, outputs=x) 
        
class ResNet50(BasedModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build_model(self):
        tf.random.set_seed(self.seed)
        inputs = Input(self.input_shape)
        # Stage 1
        X = self.convFunc(2*self.filters, (3, 3), strides = (2, 2), use_bias=self.use_bias)(inputs)
        X = self.bnFunc()(X)
        X = self.activations()(X)
        X, ind = MaxPoolingWithArgmax2D((2,2),2)(X)

        # Stage 2
        X = self.conv_block(X, self.filters*2, (3,3), stride = 1)
        X = self.identity_block(X, self.filters*2, (3,3))
        X = self.identity_block(X, self.filters*2, (3,3))
    
    
        # Stage 3 
        X = self.conv_block(X, self.filters*4, (3,3), stride = 1)
        X = self.identity_block(X, self.filters*4, (3,3))
        X = self.identity_block(X, self.filters*4, (3,3))
        
        # Stage 4 
        X = self.conv_block(X, self.filters*8, (3,3), stride = 1)
        X = self.identity_block(X, self.filters*8, (3,3))
        X = self.identity_block(X, self.filters*8, (3,3))

        
    
        # Stage 5
        # The convolutional block uses three set of filters of size [256, 256, 1024], "f" is 3, "s" is 2 and the block is "a".
        # The 5 identity blocks use three set of filters of size [256, 256, 1024], "f" is 3 and the blocks are "b", "c", "d", "e" and "f".
        X = self.conv_block(X, self.filters*16, (3,3), stride = 1)
        X = self.identity_block(X, self.filters*16, (3,3))
        X = self.identity_block(X, self.filters*16, (3,3))

        # Stage 6
        X = self.conv_block(X, self.filters*8, (3,3), stride = 1)
        X = self.identity_block(X, self.filters*8, (3,3))
        X = self.identity_block(X, self.filters*8, (3,3))
        
        # Stage 6
        X = self.conv_block(X, self.filters*4, (3,3), stride = 1)
        X = self.identity_block(X, self.filters*4, (3,3))
        X = self.identity_block(X, self.filters*4, (3,3))
        
        X = self.convFunc(2*self.filters, 3, padding='same')(X)
        X = self.bnFunc()(X)
        X = self.activations()(X)
        X = MaxUnpooling2D((2,2))([X, ind])
        X = self.bnFunc()(X)
        X = self.activations()(X)
        # Stage 6
        if self.complex_network:
            X = self.convFunc(self.filters*2, (3,3), strides=2, padding='same', transposed=True)(X)
        else:
            X = Conv2DTranspose(self.filters*2, (3,3), strides=2, padding='same')(X)
        X = self.bnFunc()(X)
        X = self.activations()(X)
        X = self.convFunc(1, 3, padding='same', use_bias=self.use_bias)(X)
        X = self.bnFunc()(X)
        X = tf.keras.layers.LeakyReLU()(X)
        return tf.keras.Model(inputs=inputs, outputs=X) 

class ResShuffle(BasedModel):
    # reference paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9062600&tag=1
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build_model(self):
        tf.random.set_seed(self.seed)
        inputs = Input(self.input_shape)
        # Stage 1
        X = self.convFunc(self.filters, (3, 3), strides = 2, padding='same', use_bias=self.use_bias)(inputs)
        X = self.bnFunc()(X)
        X = self.activations()(X)
        # Stage 2
        X = self.convFunc(2*self.filters, (3, 3), strides = 1, padding='same', use_bias=self.use_bias)(X)
        X = self.bnFunc()(X)
        X = self.activations()(X)
        X = self.convFunc(2*self.filters, (3, 3), strides = 1, padding='same', use_bias=self.use_bias)(X)
        X = self.bnFunc()(X)
        res2 = self.activations()(X)
        res2 = self.convFunc(4*self.filters, (1, 1), strides = 1, padding='same', use_bias=self.use_bias)(res2)
        # Stage 3
        X = self.convFunc(4*self.filters, (3, 3), strides = 2, padding='same', use_bias=self.use_bias)(X)
        X = self.bnFunc()(X)
        X = self.activations()(X)
        X = self.convFunc(4*self.filters, (3, 3), strides = 1, padding='same', use_bias=self.use_bias)(X)
        X = Add()([res2,X])
        X = self.bnFunc()(X)
        res3 = self.activations()(X)
        res3 = self.convFunc(8*self.filters, (1, 1), strides = 1, padding='same', use_bias=self.use_bias)(res3)
        # Stage 4
        X = self.convFunc(8*self.filters, (3, 3), strides = 2, padding='same', use_bias=self.use_bias)(X)
        X = self.bnFunc()(X)
        X = self.activations()(X)
        X = self.convFunc(8*self.filters, (3, 3), strides = 1, padding='same', use_bias=self.use_bias)(X)
        X = Add()([res3,X])
        X = self.bnFunc()(X)
        res4 = self.activations()(X)
        res4 = self.convFunc(16*self.filters, (1, 1), strides = 1, padding='same', use_bias=self.use_bias)(res4)
        # Stage 5
        X = self.convFunc(16*self.filters, (3, 3), strides = 2, padding='same', use_bias=self.use_bias)(X)
        X = self.bnFunc()(X)
        X = self.activations()(X)
        X = self.convFunc(16*self.filters, (3, 3), strides = 1, padding='same', use_bias=self.use_bias)(X)
        X = Add()([res4,X])
        X = self.bnFunc()(X)
        res5 = self.activations()(X)
        res5 = self.convFunc(32*self.filters, (1, 1), strides = 1, padding='same', use_bias=self.use_bias)(res5)
        # Stage 6
        X = self.convFunc(32*self.filters, (3, 3), strides = 1, padding='same', use_bias=self.use_bias)(X)
        X = Add()([res5, X])
        X = self.bnFunc()(X)
        X = self.activations()(X)
        X = self.convFunc(256, (3, 3), strides = 1, padding='same', use_bias=self.use_bias)(res5)
        X = PixelShuffler((16,16))(X)
        
        return tf.keras.Model(inputs=inputs, outputs=X)

class RevisedUNet(BasedModel):
    def __init__(self,filters=8,
                 size=(3,3),
                 batch_size=8,
                 lr=1e-3,
                 epochs=200,
                 seed=7414,
                 activations='FLeakyReLU',
                 losses='SSIM',
                 forward=False,
                 dropout_rate=None,
                 use_bias=False,
                 complex_network=True,
                 num_downsample_layer=5,
                 **kwargs):
        super().__init__(
            filters=filters,
            size=size,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            seed=seed,
            activations=activations,
            losses=losses,
            forward=forward,
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            complex_network=complex_network,
            num_downsample_layer=num_downsample_layer,
            **kwargs)
    
    
    def build_model(self): 
        tf.random.set_seed(self.seed)
        inputs = Input(self.input_shape)
        if self.forward:
            inputs_forward = Input(self.input_shape)            
        x = inputs
        # downsample
        inds = []
        # dropout_rates = [None]*(self.num_downsample_layer-2) + [0.5,0.9]
        dropout_rates = [None]*self.num_downsample_layer
        for ithlayer in range(self.num_downsample_layer):
            skip = self.convFunc(2**(ithlayer+1)*self.filters, (1,1), strides=2, padding='same')(x)
            x, ind = self.downsample_block_maxpool(x, 2**(ithlayer+1)*self.filters, self.size, dropout_rates[ithlayer]) # downsampling 2^num_downsample_layer
            x = Add()([x, skip])
            inds.append(ind)
        x = self.convFunc(2**(ithlayer+1)*self.filters, self.size, padding='same', use_bias=self.use_bias)(x)
        x = self.bnFunc()(x)
        x = self.activations()(x)
        # upsample
        inds = reversed(inds)
        for ind in inds:
            skip = self.convFunc(2**ithlayer*self.filters, (1,1), strides=2, padding='same', transposed=True)(x)
            x = self.upsample_block_unpool([x,ind], 2**ithlayer*self.filters, self.size, dropout_rates[ithlayer])
            x = Add()([x,skip])
            ithlayer = ithlayer - 1            
        for _ in range(3):
            x = self.convFunc(2*self.filters, 3, padding='same', use_bias=self.use_bias)(x)
            x = self.bnFunc()(x)
            x = self.activations()(x)  
        x = self.convFunc(1, 1, padding='same', use_bias=self.use_bias)(x)
        x = self.bnFunc()(x)
        if self.forward:
            x = self.activations()(x)
            x = Concatenate(axis=-1)((x, inputs_forward))
            x = self.convFunc(2, 3, padding='same', use_bias=self.use_bias)(x)
            x = self.bnFunc()(x)
            x = tf.keras.layers.LeakyReLU()(x)
            return tf.keras.Model(inputs=[inputs, inputs_forward], outputs=x)
        else:    
            x = tf.keras.layers.LeakyReLU()(x)
            return tf.keras.Model(inputs=inputs, outputs=x)
        
        

class SRN(BasedModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def res_block(self, x, filters, kernel_size):
        # reference paper: Scale-recurrent Network for Deep Image Deblurring
        # https://arxiv.org/pdf/1802.01770v1.pdf
        shortcut = x
        x = self.convFunc(filters, kernel_size, strides=1, padding='same', use_bias=self.use_bias)(x)
        x = self.bnFunc()(x)
        x = self.activations()(x)
        x = self.convFunc(filters, kernel_size, strides=1, padding='same', use_bias=self.use_bias)(x)
        x = Add()([shortcut,x])
        return x  
        
    def eblock(self, x, filters, kernel_size):
        # encoder ResBlocks
        # reference paper: Scale-recurrent Network for Deep Image Deblurring
        # https://arxiv.org/pdf/1802.01770v1.pdf
        x = self.convFunc(filters, kernel_size, strides=2, padding='same', use_bias=self.use_bias)(x)
        x = self.activations()(x)
        x = self.res_block(x, filters, kernel_size)
        x = self.res_block(x, filters, kernel_size)
        return x
        
    def build_model(self):
        tf.random.set_seed(self.seed)
        inputs = Input(self.input_shape)        
        x = inputs
        x = self.convFunc(self.filters, self.size, strides=2, padding='same', use_bias=self.use_bias)(x) # downsample
        x = self.res_block(x, self.filters, self.size)
        x = self.res_block(x, self.filters, self.size)
        x = self.res_block(x, self.filters, self.size)
        skip1 = x
        x = self.eblock(x, 2*self.filters, self.size) # downsample
        skip2 = x
        x = self.eblock(x, 4*self.filters, self.size) # downsample
        x = self.convFunc(4*self.filters, self.size, padding='same', use_bias=self.use_bias)(x)
        # upsample - decoder ResBlocks
        x = self.res_block(x, 4*self.filters, self.size)
        x = self.res_block(x, 4*self.filters, self.size)
        if self.complex_network:
            x = self.convFunc(2*self.filters, self.size, strides=2, padding='same', transposed=True)(x)
        else:
            x = Conv2DTranspose(2*self.filters, self.size, strides=2, padding='same')(x)
        x = self.activations()(x)
        # upsample - decoder ResBlocks
        x = Add()([x, skip2])
        x = self.res_block(x, 2*self.filters, self.size)
        x = self.res_block(x, 2*self.filters, self.size)
        if self.complex_network:
            x = self.convFunc(self.filters, self.size, strides=2, padding='same', transposed=True)(x)
        else:
            x = Conv2DTranspose(self.filters, self.size, strides=2, padding='same')(x)
        x = self.activations()(x)
        # last block - OutBlock
        x = Add()([x, skip1])
        x = self.res_block(x, self.filters, self.size)
        x = self.res_block(x, self.filters, self.size)
        x = self.res_block(x, self.filters, self.size)
        if self.complex_network:
            x = self.convFunc(1, self.size, strides=2, padding='same', transposed=True)(x)
        else:
            x = Conv2DTranspose(1, self.size, strides=2, padding='same')(x)
        return tf.keras.Model(inputs=inputs, outputs=x)
    
def Model(model_type):
    if model_type in {'Unet', 'UNet', 'unet'}:
        return UNet
    elif model_type in {'Segnet', 'segnet', 'SegNet'}:
        return SegNet
    elif model_type in {'PSnet', 'PSNet', 'psnet'}:
        return PSNet
    elif model_type in {'ResNet', 'resnet', 'ResNet50', 'resnet50'}:
        return ResNet50
    elif model_type in {'ResShuffle'}:
        return ResShuffle
    elif model_type in {'RevisedUNet', 'revisedUNet', 'revisedunet'}:
        return RevisedUNet
    elif model_type in {'srn', 'SRN'}:
        return SRN