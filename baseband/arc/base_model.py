# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 17:19:21 2023

@author: benzener
"""
import tensorflow as tf
import os
import numpy as np
import time
from complexnn.activation import cReLU, zReLU, modReLU, AmplitudeMaxout, FLeakyReLU, ctanh, complexReLU, mish
from complexnn.loss import ComplexRMS, ComplexMAE, ComplexMSE, SSIM, MS_SSIM, SSIM_MSE, aSSIM_MSE, ssim_map, MSE, MAE
from complexnn.conv_test import ComplexConv2D, ComplexConv1D
from complexnn.bn_test import ComplexBatchNormalization
from complexnn.pool_test import MaxPoolingWithArgmax2D, MaxUnpooling2D
from complexnn.pixelshuffle import PixelShuffler
from tensorflow.keras.layers import Input, Add, Conv2D, Conv2DTranspose, BatchNormalization, Dropout, Concatenate, LeakyReLU, MaxPool2D
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.activations import tanh
from datetime import datetime
import matplotlib.pyplot as plt
if __name__ == '__main__':
    import sys
    currentpath = os.getcwd()
    addpath = os.path.dirname(os.path.dirname(currentpath))
    if addpath not in sys.path:
        sys.path.append(addpath)
    from baseband.setting import constant
    from baseband.utils.fig_utils import envelope_fig
    from baseband.utils.data_utils import normalization
    from baseband.model_utils import InOne, CosineDecay, ComplexNormalization, ComplexConcatenate, GlobalAveragePooling
    sys.path.remove(addpath)
else:
    from ..setting import constant
    from ..utils.fig_utils import envelope_fig
    from ..utils.data_utils import normalization
    from .model_utils import InOne, CosineDecay, ComplexNormalization, ComplexConcatenate, GlobalAveragePooling


class BaseModel(): 
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
        self.concatFunc = ComplexConcatenate if self.complex_network else Concatenate
        self.fine_tune = False
           
    def downsample_block_conv(self, x, filters, kernel_size, dropout_rate=None, bn=False, normalize_weight=False, constraints=None, kernel_initializer='complex_independent'):
        x = self.convFunc(filters, kernel_size, padding='same', use_bias=self.use_bias, normalize_weight=normalize_weight, kernel_constraint=constraints, kernel_initializer=kernel_initializer)(x)
        if bn:
            x = self.bnFunc()(x)
        x = self.activations()(x)
        x = self.convFunc(filters, kernel_size, strides=2, padding='same', use_bias=self.use_bias, normalize_weight=normalize_weight, kernel_constraint=constraints, kernel_initializer=kernel_initializer)(x)
        if bn:
            x = self.bnFunc()(x)
        x = self.activations()(x)
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
        return x
    
    def downsample_block_argmax(self, x, filters, kernel_size, dropout_rate=None, bn=False, normalize_weight=False, constraints=None, kernel_initializer='complex_independent'):
        x = self.convFunc(filters, kernel_size, padding='same', use_bias=self.use_bias, normalize_weight=normalize_weight, kernel_constraint=constraints, kernel_initializer=kernel_initializer)(x)
        if bn:
            x = self.bnFunc()(x)
        x = self.activations()(x)
        x, ind = MaxPoolingWithArgmax2D((2,2),2)(x)
        if bn:
            x = self.bnFunc()(x)
        x = self.activations()(x)
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
        return x, ind
    
    def downsample_block_maxpool(self, x, filters, kernel_size, dropout_rate=None, bn=False, normalize_weight=False, constraints=None, kernel_initializer='complex_independent'):
        x = self.convFunc(filters, kernel_size, padding='same', use_bias=self.use_bias, normalize_weight=normalize_weight, kernel_constraint=constraints, kernel_initializer=kernel_initializer)(x)
        if bn:
            x = self.bnFunc()(x)
        x = self.activations()(x)
        x = tf.keras.layers.MaxPool2D((2,2),2,'same')(x)
        if bn:
            x = self.bnFunc()(x)
        x = self.activations()(x)
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
        return x
    
    def upsample_block_PS(self, x, filters, kernel_size, dropout_rate=None, bn=False, normalize_weight=False, constraints=None, kernel_initializer='complex_independent'):
        # pixel shuffle
        x = PixelShuffler((2,2))(x)
        if bn:
            x = self.bnFunc()(x)
        x = self.activations()(x)
        x = self.convFunc(filters, kernel_size, padding='same', use_bias=self.use_bias, normalize_weight=normalize_weight, kernel_constraint=constraints, kernel_initializer=kernel_initializer)(x)
        if bn:
            x = self.bnFunc()(x)
        x = self.activations()(x)
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
        return x

    def upsample_block_transp(self, x, filters, kernel_size, dropout_rate=None, bn=False, normalize_weight=False, constraints=None, kernel_initializer='complex_independent'):
        if self.complex_network:
            x = self.convFunc(filters, kernel_size, strides=2, padding='same', transposed=True, normalize_weight=normalize_weight, kernel_constraint=constraints, kernel_initializer=kernel_initializer)(x)
        else:
            x = Conv2DTranspose(filters, kernel_size, strides=2, padding='same', normalize_weight=normalize_weight, kernel_constraint=constraints)(x)
        if bn:
            x = self.bnFunc()(x)
        x = self.activations()(x)
        x = self.convFunc(filters, kernel_size, padding='same', use_bias=self.use_bias, normalize_weight=normalize_weight, kernel_constraint=constraints)(x)
        if bn:
            x = self.bnFunc()(x)
        x = self.activations()(x)
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
        return x
    
    def upsample_block_unpool(self, x, filters, kernel_size, dropout_rate=None, bn=False, normalize_weight=False, constraints=None, kernel_initializer='complex_independent'):
        x = MaxUnpooling2D((2,2))(x)
        if bn:
            x = self.bnFunc()(x)
        x = self.activations()(x)
        x = self.convFunc(filters, kernel_size, padding='same', use_bias=self.use_bias, normalize_weight=normalize_weight, kernel_constraint=constraints, kernel_initializer=kernel_initializer)(x)
        if bn:
            x = self.bnFunc()(x)
        x = self.activations()(x)
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
        return x
    
    def upsample_block_interp(self, x, filters, kernel_size, dropout_rate=None, bn=False, normalize_weight=False, constraints=None, kernel_initializer='complex_independent'):
        x = tf.keras.layers.UpSampling2D((2,2), interpolation='bilinear')(x)
        if bn:
            x = self.bnFunc()(x)
        x = self.activations()(x)
        x = self.convFunc(filters, kernel_size, padding='same', use_bias=self.use_bias, normalize_weight=normalize_weight, kernel_constraint=constraints, kernel_initializer=kernel_initializer)(x)
        if bn:
            x = self.bnFunc()(x)
        x = self.activations()(x)
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
        return x
    
    def identity_block(self, x, filters, kernel_size, dropout_rate=None, bn=False, normalize_weight=False):
        last_layer_filters = x.shape[-1]//2 if self.complex_network else x.shape[-1]
        shortcut = x
        x = self.convFunc(filters, (1,1), padding='same', strides=1, use_bias=self.use_bias, normalize_weight=normalize_weight)(x)
        if bn:
            x = self.bnFunc()(x)
        x = self.activations()(x)
        x = self.convFunc(filters, kernel_size, padding='same', strides=1, use_bias=self.use_bias, normalize_weight=normalize_weight)(x)
        if bn:
            x = self.bnFunc()(x)
        x = self.activations()(x)
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
        x = self.convFunc(last_layer_filters, (1,1), padding='same', strides=1, use_bias=self.use_bias, normalize_weight=normalize_weight)(x)
        if bn:
            x = self.bnFunc()(x)
        x = self.activations()(x)
        x = Add()([x, shortcut])
        x = self.activations()(x)
        return x
    
    def conv_block(self, x, filters, kernel_size, stride, dropout_rate=None, bn=False, normalize_weight=False):
        last_layer_filters = x.shape[-1]//2 if self.complex_network else x.shape[-1]
        shortcut = x
        x = self.convFunc(filters, (1,1), padding='same', strides=stride, use_bias=self.use_bias, normalize_weight=normalize_weight)(x)
        if bn:
            x = self.bnFunc()(x)
        x = self.activations()(x)
        x = self.convFunc(filters, kernel_size, padding='same', strides=1, use_bias=self.use_bias, normalize_weight=normalize_weight)(x)
        if bn:
            x = self.bnFunc()(x)
        x = self.activations()(x)
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
        x = self.convFunc(last_layer_filters, (1,1), padding='same', strides=1, use_bias=self.use_bias, normalize_weight=normalize_weight)(x)
        if bn:
            x = self.bnFunc()(x)
        x = self.activations()(x)
        shortcut = self.convFunc(last_layer_filters, (1,1), strides=stride, padding='same', use_bias=self.use_bias, normalize_weight=normalize_weight)(shortcut)
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
                if self.activations not in {'cReLU', 'zReLU', 'modReLU', 'ctanh','LeakyReLU', 'AMU', 'FLeakyReLU', 'complexReLU', 'mish'}:
                    raise  KeyError('Unsupported activation')
        else:
            assert self.input_shape[-1] == 1
            if isinstance(self.losses, str):
                if self.losses not in {'MSE','SSIM', 'MAE','MS_SSIM', 'SSIM_MSE', 'aSSIM_MSE', 'SSIM_map'}:
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
            'MSE':MSE,
            'MAE':MAE,
            'ctanh':ctanh,
            'FLeakyReLU': FLeakyReLU,
            'AmplitudeMaxout':AmplitudeMaxout,
            'complexReLU':complexReLU,
            'mish':mish
            }
        return custom_object
    
    def _generate_name(self):
        now = datetime.now()
        day_month_year = now.strftime("%d%m%Y")
        data_type = 'complex' if self.complex_network else 'real'
        forward = 'forward' if self.forward else 'Notforward'
        model_name = self.__class__.__name__
        epochs = str(self.epochs)
        loss_name = self.losses if isinstance(self.losses,str) else self.losses.__name__
        act_name = self.activations if isinstance(self.activations,str) else self.activations.__name__
        opt_name = self.optimizer.__class__.__name__
        if self.fine_tune:
            self.model_name = f'{data_type}{model_name}_{forward}_{epochs}_{loss_name}_{act_name}_{opt_name}_finetune' + day_month_year
        else:
            self.model_name = f'{data_type}{model_name}_{forward}_{epochs}_{loss_name}_{act_name}_{opt_name}' + day_month_year
    
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
            'activation':self.activations if isinstance(self.activations,str) else self.activations.__name__,
            'loss':self.losses if isinstance(self.losses,str) else self.losses.__name__,
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
        if self.activations in {'cReLU', 'zReLU', 'modReLU', 'ctanh','LeakyReLU', 'AMU', 'FLeakyReLU', 'complexReLU', 'mish'}:
            self.activations = {
                'cReLU': cReLU,
                'zReLU': zReLU,
                'modReLU': modReLU,
                'ctanh': tanh,
                'LeakyReLU': LeakyReLU,
                'FLeakyReLU': FLeakyReLU,
                'AMU'      : AmplitudeMaxout,
                'complexReLU': complexReLU,
                'mish': mish
                }[self.activations]
        else:
            if isinstance(self.activations, str):
                raise KeyError('activation function is not defined')
            # if isinstance(self.activations, type):
            #     self.activations = self.activations()

    def _convert_loss(self):
        # determine loss function 
        if self.losses in {'ComplexRMS', 'ComplexMAE', 'ComplexMSE', 'SSIM', 'MSE','MAE', 'MS_SSIM' ,'SSIM_MSE', 'aSSIM_MSE', 'SSIM_map'}:
            self.losses = {
                'ComplexRMS': ComplexRMS,
                'ComplexMAE': ComplexMAE,
                'ComplexMSE': ComplexMSE,
                'SSIM'      : SSIM,
                'MS_SSIM'   : MS_SSIM,
                'SSIM_MSE'  : SSIM_MSE,
                'MSE'       : MSE,
                'MAE'       : MAE,
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
        x_train, y_train = normalization(x), normalization(y)
        
        if self.validation_data is not None:
            x_val, y_val = self.validation_data
            x_val, y_val = normalization(x_val), normalization(y_val)
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
            self._convert_loss()
            self._convert_act()
            self.model = self.build_model()
        else:
            self.model_info['basedmodel'] = model_name
            # self.model = tf.keras.models.load_model(os.path.join(constant.MODELPATH,model_name,model_name+'.h5'),self.get_custom_object(), False)
            self.model = tf.keras.models.load_model(os.path.join(constant.MODELPATH,model_name,model_name),self.get_custom_object(), False)
            self.fine_tune = True
            self._convert_loss()
            self._convert_act()

        
        self.model.summary()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, amsgrad=False)
        # self.optimizer = tf.keras.optimizers.Adadelta(self.lr)
        # self.optimizer = AMSGrad(learning_rate=self.lr)
        # self.optimizer = tf.keras.optimizers.SGD(CosineDecay(self.lr, 20, 1e-2), momentum=0.5, nesterov=True)
        # self.optimizer = tf.keras.optimizers.RMSprop(self.lr,momentum=0.9)
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
            # plt.figure()
            # plt.plot(y_val[5])
            # plt.show()
        except Exception:
            no_validation = True
        envelope_fig(y_train[4:5],60, 0, 
                      "Ground truth", 
                      saved_name="Ground_truth",
                      model_name=self.model_name,
                      saved_dir='PredictionEachEpoch')
        # plt.figure()
        # plt.plot(y_train[4])
        # plt.show()
        
        
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
                # if tf.reduce_any(tf.math.is_nan(self.model.weights[2])):
                #     print(step, epoch)
                #     raise ValueError('Nan')
                # else:
                    # print(self.model.weights)
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
            # if np.mean(loss_train_epoch) < 0.003:
            #     train_img = self.model.predict(x_train[:8])
            #     envelope_fig(normalization(train_img[4:5]),60, 0, 
            #                  f"Prediction-epoch-{epoch+1}", 
            #                  saved_name=f"epoch-{epoch+1}",
            #                  model_name=self.model_name,
            #                  saved_dir='PredictionEachEpoch')
            #     envelope_fig(normalization(train_img[5:6]),60, 0, 
            #                  f"Prediction5-epoch-{epoch+1}")
            #     return self.model, history
            if (epoch+1)%5 == 0:
                train_img = self.model.predict(x_train[:8])
                print(np.min(np.abs(train_img)), np.mean(np.abs(train_img)), np.max(np.abs(train_img)))
                envelope_fig(train_img[4:5],60, 0, 
                              f"Prediction-epoch-{epoch+1}", 
                              saved_name=f"epoch-{epoch+1}",
                              model_name=self.model_name,
                              saved_dir='PredictionEachEpoch')
                # plt.figure()
                # plt.plot(train_img[4], label='prediction')
                # plt.plot(y_train[4], label='ground truth')
                # plt.title(f"Prediction-epoch-{epoch+1}")
                # plt.show()
                if not no_validation:
                    val_img = self.model.predict(x_val[:8])
                    envelope_fig(val_img[5:6],60, 0, 
                                  f"Validation-epoch-{epoch+1}", 
                                  saved_name=f"epoch-val-{epoch+1}",
                                  model_name=self.model_name,
                                  saved_dir='PredictionEachEpoch')
                    # plt.figure()
                    # plt.plot(val_img[5], label='prediction')
                    # plt.plot(y_val[5], label='ground truth')
                    # plt.title(f"Validation-epoch-{epoch+1}")
                    # plt.show()
            
            if (epoch+1)%500 == 0:
                print('saving...')
                save_path = os.path.join(constant.MODELPATH, self.model_name, f'checkpoints/chkpt-epoch{epoch+1:04d}/')
                if os.path.exists(save_path):
                    os.makedirs(save_path)
                self.model.save_weights(save_path)
        return self.model, history
    
    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            result = self.model(x, training=True)
            loss_value = self.losses(y, result)
        # tf.print(tf.reduce_max(tf.abs(result)),tf.reduce_mean(tf.abs(result)))
        # tf.print('gg',gg,'cc',cc, 'out', result)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        # tf.print(tf.reduce_mean(grads[-2]))
        return loss_value
    
    @tf.function
    def test_step(self, x, y):
        val_result = self.model(x, training=False)
        return self.losses(y, val_result)
    
    def training_multiple(self, x, y):
        '''
            without supporting Forward model
        '''
        
        history = {
            'loss_optimizer1':[],
            'loss_optimizer2':[]
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

        self._generate_name()
        self._convert_loss()
        self._convert_act()
        self.model1 = self.build_model()
        self.model2 = self.build_model()


        try:
            N = x_val.shape[0]
            valid_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).shuffle(N).batch(self.batch_size)
            no_validation = False
            history['val_loss_optimizer1'] = []
            history['val_loss_optimizer2'] = []
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
        
        
        self.model1.summary()
        self.optimizer1 = tf.keras.optimizers.Adam(learning_rate=self.lr)
        # self.optimizer2 = tf.keras.optimizers.SGD(
        #     tf.keras.optimizers.schedules.PiecewiseConstantDecay([self.epochs//2],[self.lr*100,self.lr*10]),
        #     momentum=0.99,
        #     nesterov=True)
        for epoch in range(self.epochs):
            try:
                self.losses = self.losses(epoch/self.epochs)
            except TypeError:
                pass
            if self.batchsizeschedule and epoch != 0:
                batch_size = int(1 + self.batch_size//self.epochs)
                train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(N).batch(batch_size)
            s = time.time()
            loss_train_epoch1 = []
            loss_train_epoch2 = []
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                losses = self.train_step_multiple(x_batch_train, y_batch_train)
                loss_train_epoch1.append(losses[0])
                loss_train_epoch2.append(losses[1])
            history['loss_optimizer1'].append(np.mean(loss_train_epoch1))
            history['loss_optimizer2'].append(np.mean(loss_train_epoch2))
            if no_validation:
                e = time.time()
                print(f'Epoch:{epoch+1}/{self.epochs} - {(e-s):.1f}s - ' \
                      f'loss1:{np.mean(loss_train_epoch1):.6f} - '\
                          f'loss2:{np.mean(loss_train_epoch2):.6f} \n')
            else:
                loss_valid_epoch1 = []
                loss_valid_epoch2 = []
                for x_batch_val, y_batch_val in valid_dataset:
                    losses = self.test_step_multiple(x_batch_val, y_batch_val)
                    loss_valid_epoch1.append(losses[0])
                    loss_valid_epoch2.append(losses[1])
                history['val_loss_optimizer1'].append(np.mean(loss_valid_epoch1))
                history['val_loss_optimizer2'].append(np.mean(loss_valid_epoch2))
                e = time.time()
                print(f'Epoch:{epoch+1}/{self.epochs} - {(e-s):.1f}s - ' \
                      f'loss1:{np.mean(loss_train_epoch1):.6f} - ' \
                      f'loss2:{np.mean(loss_train_epoch2):.6f} - ' \
                      f'val_loss1:{np.mean(loss_valid_epoch1):.6f} - ' \
                      f'val_loss2:{np.mean(loss_valid_epoch2):.6f} \n')
            if (epoch+1)%10 == 0:
                train_img1 = self.model1.predict(x_train[:8])
                train_img2 = self.model2.predict(x_train[:8])
                envelope_fig(normalization(train_img1[4:5]),60, 0, 
                             f"Prediction1-epoch-{epoch+1}", 
                             saved_name=f"optimizer1-epoch-{epoch+1}",
                             model_name=self.model_name,
                             saved_dir='PredictionEachEpoch')
                envelope_fig(normalization(train_img2[4:5]),60, 0, 
                             f"Prediction2-epoch-{epoch+1}", 
                             saved_name=f"optimizer2-epoch-{epoch+1}",
                             model_name=self.model_name,
                             saved_dir='PredictionEachEpoch')
                if not no_validation:
                    val_img1 = self.model1.predict(x_val[:8])
                    val_img2 = self.model2.predict(x_val[:8])
                    envelope_fig(normalization(val_img1[5:6]),60, 0, 
                                 f"Validation1-epoch-{epoch+1}", 
                                 saved_name=f"optimizer1-epoch-val-{epoch+1}",
                                 model_name=self.model_name,
                                 saved_dir='PredictionEachEpoch')
                    envelope_fig(normalization(val_img2[5:6]),60, 0, 
                                 f"Validation2-epoch-{epoch+1}", 
                                 saved_name=f"optimizer1-epoch-val-{epoch+1}",
                                 model_name=self.model_name,
                                 saved_dir='PredictionEachEpoch')
            
            if (epoch+1)%500 == 0:
                print('saving...')
                save_path = os.path.join(constant.MODELPATH, self.model_name, f'checkpoints/chkpt-epoch{epoch+1:04d}/')
                if os.path.exists(save_path):
                    os.makedirs(save_path)
                self.model1.save_weights(save_path)
        return self.model1, self.model2, history
    
    @tf.function
    def train_step_multiple(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            result1 = self.model1(x, training=True)
            result2 = self.model2(x, training=True)
            loss_value1 = self.losses(y, result1)
            loss_value2 = self.losses(y, result2)
        grads1 = tape.gradient(loss_value1, self.model1.trainable_weights)
        self.optimizer1.apply_gradients(zip(grads1, self.model1.trainable_weights))
        grads2 = tape.gradient(loss_value2, self.model2.trainable_weights)
        self.optimizer2.apply_gradients(zip(grads2, self.model2.trainable_weights))
        return [loss_value1, loss_value2]
    
    @tf.function
    def test_step_multiple(self, x, y):
        val_result1 = self.model1(x, training=False)
        val_result2 = self.model2(x, training=False)
        return [self.losses(y, val_result1), self.losses(y, val_result2)]
    
class UNet(BaseModel):
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
        skips = reversed(skips[1:])
        for skip in skips:
            x = self.concatFunc()([x, skip])
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

class SegNet(BaseModel):
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
        if self.dropout_rate is not None:
            dropout_rates = [None]*(self.num_downsample_layer-2) + [0.5*self.dropout_rate, self.dropout_rate]
        for ithlayer in range(1,self.num_downsample_layer+1):
            x, ind = self.downsample_block_argmax(x, 2**(ithlayer+1)*self.filters, self.size, dropout_rate=dropout_rates[ithlayer]) # downsampling 2^num_downsample_layer
            inds.append(ind)
        x = self.convFunc(2**(ithlayer+1)*self.filters, self.size, padding='same', use_bias=self.use_bias)(x)
        x = self.bnFunc()(x)
        x = self.activations()(x)
        # upsample
        inds = reversed(inds)
        for ind in inds:
            x = self.upsample_block_unpool([x,ind], 2**ithlayer*self.filters, self.size, dropout_rate=dropout_rates[ithlayer])
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
        
class PSNet(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build_model(self):
        tf.random.set_seed(self.seed)
        inputs = Input(self.input_shape)
        if self.forward:
            inputs_forward = Input(self.input_shape)            
        x = inputs
        # downsample
        for ithlayer in range(self.num_downsample_layer):
            x = self.downsample_block_conv(x, 4**(ithlayer+1)*self.filters, self.size) # downsampling 2^num_downsample_layer
        x = self.convFunc(2**(ithlayer+1)*self.filters, self.size, padding='same', use_bias=self.use_bias)(x)
        x = self.bnFunc()(x)
        x = self.activations()(x)
        # upsample
        for ithlayer in range(self.num_downsample_layer,0,-1):
            x = self.upsample_block_PS(x, 4**(ithlayer+1)*self.filters, self.size)  
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
        
class ResNet50(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build_model(self):
        tf.random.set_seed(self.seed)
        inputs = Input(self.input_shape)
        # Stage 1
        X = self.convFunc(2*self.filters, (3, 3), strides = (2, 2), use_bias=self.use_bias)(inputs)
        X = self.bnFunc()(X)
        X = self.activations(X)
        X, ind = MaxPoolingWithArgmax2D((2,2),2)(X)

        # Stage 2
        X = self.conv_block(X, self.filters*2, (3,3), stride = 1)
        X = self.identity_block(X, self.filters*2, (3,3))
        X = self.identity_block(X, self.filters*2, (3,3), bn=True)
    
    
        # Stage 3 
        X = self.conv_block(X, self.filters*4, (3,3), stride = 1)
        X = self.identity_block(X, self.filters*4, (3,3))
        X = self.identity_block(X, self.filters*4, (3,3), bn=True)
        
        # Stage 4 
        X = self.conv_block(X, self.filters*8, (3,3), stride = 1)
        X = self.identity_block(X, self.filters*8, (3,3))
        X = self.identity_block(X, self.filters*8, (3,3), bn=True)

        # Stage 5
        # The convolutional block uses three set of filters of size [256, 256, 1024], "f" is 3, "s" is 2 and the block is "a".
        # The 5 identity blocks use three set of filters of size [256, 256, 1024], "f" is 3 and the blocks are "b", "c", "d", "e" and "f".
        X = self.conv_block(X, self.filters*16, (3,3), stride = 1)
        X = self.identity_block(X, self.filters*16, (3,3))
        X = self.identity_block(X, self.filters*16, (3,3), bn=True)

        # Stage 6
        X = self.conv_block(X, self.filters*32, (3,3), stride = 1)
        X = self.identity_block(X, self.filters*32, (3,3))
        X = self.identity_block(X, self.filters*32, (3,3), bn=True)
        
        # Stage 6
        X = self.conv_block(X, self.filters*64, (3,3), stride = 1)
        X = self.identity_block(X, self.filters*64, (3,3))
        X = self.identity_block(X, self.filters*64, (3,3), bn=True)
        
        X = self.convFunc(2*self.filters, 3, padding='same')(X)
        X = self.activations(X)
        X = MaxUnpooling2D((2,2))([X, ind])
        X = self.activations(X)
        # Stage 6
        if self.complex_network:
            X = self.convFunc(self.filters*2, (3,3), strides=2, padding='same', transposed=True)(X)
        else:
            X = Conv2DTranspose(self.filters*2, (3,3), strides=2, padding='same')(X)
        X = self.activations(X)
        X = self.convFunc(1, 3, padding='same', use_bias=self.use_bias)(X)
        X = tf.keras.layers.LeakyReLU()(X)
        return tf.keras.Model(inputs=inputs, outputs=X) 

class ResShuffle(BaseModel):
    # reference paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9062600&tag=1
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build_model(self):
        tf.random.set_seed(self.seed)
        inputs = Input(self.input_shape)
        # Stage 1
        X = self.convFunc(self.filters, (3, 3), strides = 2, padding='same', use_bias=self.use_bias)(inputs)
        X = self.bnFunc()(X)
        X = self.activations(X)
        # Stage 2
        X = self.convFunc(2*self.filters, (3, 3), strides = 1, padding='same', use_bias=self.use_bias)(X)
        X = self.bnFunc()(X)
        X = self.activations(X)
        X = self.convFunc(2*self.filters, (3, 3), strides = 1, padding='same', use_bias=self.use_bias)(X)
        X = self.bnFunc()(X)
        res2 = self.activations(X)
        res2 = self.convFunc(4*self.filters, (1, 1), strides = 1, padding='same', use_bias=self.use_bias)(res2)
        # Stage 3
        X = self.convFunc(4*self.filters, (3, 3), strides = 2, padding='same', use_bias=self.use_bias)(X)
        X = self.bnFunc()(X)
        X = self.activations(X)
        X = self.convFunc(4*self.filters, (3, 3), strides = 1, padding='same', use_bias=self.use_bias)(X)
        X = Add()([res2,X])
        X = self.bnFunc()(X)
        res3 = self.activations(X)
        res3 = self.convFunc(8*self.filters, (1, 1), strides = 1, padding='same', use_bias=self.use_bias)(res3)
        # Stage 4
        X = self.convFunc(8*self.filters, (3, 3), strides = 2, padding='same', use_bias=self.use_bias)(X)
        X = self.bnFunc()(X)
        X = self.activations(X)
        X = self.convFunc(8*self.filters, (3, 3), strides = 1, padding='same', use_bias=self.use_bias)(X)
        X = Add()([res3,X])
        X = self.bnFunc()(X)
        res4 = self.activations(X)
        res4 = self.convFunc(16*self.filters, (1, 1), strides = 1, padding='same', use_bias=self.use_bias)(res4)
        # Stage 5
        X = self.convFunc(16*self.filters, (3, 3), strides = 2, padding='same', use_bias=self.use_bias)(X)
        X = self.bnFunc()(X)
        X = self.activations(X)
        X = self.convFunc(16*self.filters, (3, 3), strides = 1, padding='same', use_bias=self.use_bias)(X)
        X = Add()([res4,X])
        X = self.bnFunc()(X)
        res5 = self.activations(X)
        res5 = self.convFunc(32*self.filters, (1, 1), strides = 1, padding='same', use_bias=self.use_bias)(res5)
        # Stage 6
        X = self.convFunc(32*self.filters, (3, 3), strides = 1, padding='same', use_bias=self.use_bias)(X)
        X = Add()([res5, X])
        X = self.bnFunc()(X)
        X = self.activations(X)
        X = self.convFunc(256, (3, 3), strides = 1, padding='same', use_bias=self.use_bias)(res5)
        X = PixelShuffler((16,16))(X)
        
        return tf.keras.Model(inputs=inputs, outputs=X)

class RevisedSegNet(BaseModel):
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
        if self.dropout_rate is not None:
            dropout_rates = [None]*(self.num_downsample_layer-2) + [0.5*self.dropout_rate, self.dropout_rate]
        else:
            dropout_rates = [None]*self.num_downsample_layer
        for ithlayer in range(self.num_downsample_layer):
            if ithlayer > self.num_downsample_layer - 2:
                x, ind = self.downsample_block_argmax(x, 2**(ithlayer+1)*self.filters, self.size, dropout_rate=self.dropout_rate, bn=True) # downsampling 2^num_downsample_layer
            else:
                x, ind = self.downsample_block_argmax(x, 2**(ithlayer+1)*self.filters, self.size, dropout_rate=dropout_rates[ithlayer]) # downsampling 2^num_downsample_layer
            inds.append(ind)
        x = self.convFunc(2**(ithlayer+1)*self.filters, self.size, padding='same', use_bias=self.use_bias)(x)
        x = self.bnFunc()(x)
        x = self.activations()(x)
        # upsample
        inds = reversed(inds)
        for ind in inds:
            if ithlayer > self.num_downsample_layer - 2:
                x = self.upsample_block_unpool([x,ind], 2**ithlayer*self.filters, self.size, dropout_rate=dropout_rates[ithlayer], bn=True)
            else:
                x = self.upsample_block_unpool([x,ind], 2**ithlayer*self.filters, self.size, dropout_rate=dropout_rates[ithlayer])
            ithlayer = ithlayer - 1
        for _ in range(3):
            x = self.convFunc(2*self.filters, 3, padding='same', use_bias=self.use_bias)(x)
            x = self.activations()(x)  
        x = self.convFunc(1, (1,1), padding='same', use_bias=self.use_bias)(x)
        if self.forward:
            x = self.activations()(x)
            x = Concatenate(axis=-1)((x, inputs_forward))
            x = self.convFunc(2, 3, padding='same', use_bias=self.use_bias)(x)
            x = tf.keras.layers.LeakyReLU()(x)
            return tf.keras.Model(inputs=[inputs, inputs_forward], outputs=x)
        else:    
            x = tf.keras.layers.LeakyReLU()(x)
            return tf.keras.Model(inputs=inputs, outputs=x)
        
        
class RevisedUNet(BaseModel):
    def __init__(self,
                 filters=16,
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
        x = self.convFunc(4*self.filters, self.filters, strides=1, padding='same')(x) # 32
        # x = self.bnFunc()(x)
        x = self.activations()(x)
        # downsample - 1
        x = self.convFunc(4*self.filters, self.filters, strides=2, padding='same')(x) # 32
        # x = self.bnFunc()(x)
        x = self.activations()(x)
        x = self.convFunc(8*self.filters, self.filters, strides=1, padding='same')(x) # 64
        # x = self.bnFunc()(x)
        x = self.activations()(x)
        # downsample - 2
        x = self.convFunc(8*self.filters, self.filters, strides=2, padding='same')(x) # 64
        # x = self.bnFunc()(x)
        x = self.activations()(x)
        x = self.convFunc(16*self.filters, self.filters, strides=1, padding='same')(x) # 128
        x = self.bnFunc()(x)
        x = self.activations()(x)
        # downsample - 3
        x = self.convFunc(16*self.filters, self.filters, strides=2, padding='same')(x) # 128
        # x = self.bnFunc()(x)
        x = self.activations()(x)
        skip1 = x
        x = self.convFunc(32*self.filters, self.filters, strides=1, padding='same')(x) # 256
        x = self.bnFunc()(x)
        x = self.activations()(x)
        # downsample - 4
        x = self.convFunc(32*self.filters, self.filters, strides=2, padding='same')(x) # 256
        # x = self.bnFunc()(x)
        x = self.activations()(x)
        skip2 = x
        x = self.convFunc(64*self.filters, self.filters, strides=1, padding='same')(x) # 512
        x = self.bnFunc()(x)
        x = self.activations()(x)
        # downsample - 5
        x = self.convFunc(64*self.filters, self.filters, strides=2, padding='same')(x) # 512
        # x = self.bnFunc()(x)
        x = self.activations()(x)
        skip3 = x
        x = self.convFunc(64*self.filters, self.filters, strides=1, padding='same')(x) # 512
        x = self.bnFunc()(x)
        x = self.activations()(x)
        # upsample - 5
        x = self.concatFunc()([x,skip3])
        x = self.convFunc(self.filters, self.filters, strides=2, padding='same', transposed=True)(x) # 8
        # x = self.bnFunc()(x)
        x = self.activations()(x)
        x = self.convFunc(self.filters, self.filters, strides=1, padding='same')(x) # 8
        x = self.bnFunc()(x)
        x = self.activations()(x)
        # upsample - 4
        x = self.concatFunc()([x,skip2])
        x = self.convFunc(2*self.filters, self.filters, strides=2, padding='same', transposed=True)(x) # 16
        # x = self.bnFunc()(x)
        x = self.activations()(x)
        x = self.convFunc(2*self.filters, self.filters, strides=1, padding='same')(x) # 8
        x = self.bnFunc()(x)
        x = self.activations()(x)
        # upsample - 3
        x = self.concatFunc()([x,skip1])
        x = self.convFunc(4*self.filters, self.filters, strides=2, padding='same', transposed=True)(x) # 16
        # x = self.bnFunc()(x)
        x = self.activations()(x)
        x = self.convFunc(4*self.filters, self.filters, strides=1, padding='same')(x) # 8
        x = self.bnFunc()(x)
        x = self.activations()(x)
        # upsample - 2
        x = self.convFunc(8*self.filters, self.filters, strides=2, padding='same', transposed=True)(x) # 16
        # x = self.bnFunc()(x)
        x = self.activations()(x)
        x = self.convFunc(8*self.filters, self.filters, strides=1, padding='same')(x) # 8
        x = self.bnFunc()(x)
        x = self.activations()(x)
        # upsample - 1
        x = self.convFunc(16*self.filters, self.filters, strides=2, padding='same', transposed=True)(x) # 16
        # x = self.bnFunc()(x)
        x = self.activations()(x)
        x = self.convFunc(16*self.filters, self.filters, strides=1, padding='same')(x) # 8
        # x = self.bnFunc()(x)
        x = self.activations()(x)
        for _ in range(3):
            x = self.convFunc(2*self.filters, self.filters, strides=1, padding='same')(x) # 8
            x = self.bnFunc()(x)
            x = self.activations()(x)
        x = self.convFunc(1, (1,1), padding='same', use_bias=self.use_bias)(x)
        if self.forward:
            x = self.activations()(x)
            x = Concatenate(axis=-1)((x, inputs_forward))
            x = self.convFunc(2, 3, padding='same', use_bias=self.use_bias)(x)
            x = tf.keras.layers.LeakyReLU()(x)
            return tf.keras.Model(inputs=[inputs, inputs_forward], outputs=x)
        else:    
            x = self.activations()(x)
            return tf.keras.Model(inputs=inputs, outputs=x)
        
class ResUNet(BaseModel):
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
    def _downsample_block_argmax(self, x, filters, kernel_size, dropout_rate=None, bn=False, normalize_weight=False, constraints=None, kernel_initializer='complex_independent'):
        x = self.convFunc(filters, kernel_size, padding='same', use_bias=self.use_bias, normalize_weight=normalize_weight, kernel_constraint=constraints, kernel_initializer=kernel_initializer)(x)
        x = self.activations()(x)
        x, ind = MaxPoolingWithArgmax2D((2,2),2)(x)
        if bn:
            x = self.bnFunc()(x)
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
        return x, ind
    def _upsample_block_unpool(self, x, filters, kernel_size, dropout_rate=None, bn=False, normalize_weight=False, constraints=None, kernel_initializer='complex_independent'):
        x, ind = x
        x = self.convFunc(filters, kernel_size, padding='same', use_bias=self.use_bias, normalize_weight=normalize_weight, kernel_constraint=constraints, kernel_initializer=kernel_initializer)(x)
        x = self.activations()(x)
        x = MaxUnpooling2D((2,2))([x,ind])
        if bn:
            x = self.bnFunc()(x)
        
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
        return x
    
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
            skip = self.activations(skip)
            x, ind = self._downsample_block_argmax(x, 2**(ithlayer+1)*self.filters, self.size, dropout_rates[ithlayer]) # downsampling 2^num_downsample_layer
            x = Add()([x, skip])
            inds.append(ind)
        x = self.convFunc(2**(ithlayer+1)*self.filters, self.size, padding='same', use_bias=self.use_bias)(x)
        x = self.bnFunc()(x)
        x = self.activations()(x)
        # upsample
        inds = reversed(inds)
        for ind in inds:
            skip = self.convFunc(2**(ithlayer+1)*self.filters, (1,1), strides=2, padding='same', transposed=True)(x)
            skip = self.activations(skip)
            x = self._upsample_block_unpool([x,ind], 2**(ithlayer+1)*self.filters, self.size, dropout_rates[ithlayer])
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
        
        

class SRN(BaseModel):
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
    

    
class TestUNet(BaseModel):
    def __init__(self, filters=4,
                 size=(3,3),
                 batch_size=32,
                 lr=1e-3,
                 epochs=1000,
                 seed=7414,
                 activations='mish',
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
        
    def _downsample_block_conv(self, x, filters, kernel_size, dropout_rate=None, bn=False, normalize_weight=False, constraints=None, kernel_initializer='complex_independent'):
        x = self.convFunc(filters, kernel_size, padding='same', use_bias=self.use_bias, normalize_weight=normalize_weight, kernel_constraint=constraints, kernel_initializer=kernel_initializer)(x)
        x = self.activations()(x)
        # x = self.convFunc(filters, kernel_size, padding='same', use_bias=self.use_bias, normalize_weight=normalize_weight, kernel_constraint=constraints, kernel_initializer=kernel_initializer)(x)
        # x = self.activations()(x)
        x = self.convFunc(filters, kernel_size, strides=2, padding='same', use_bias=self.use_bias, normalize_weight=normalize_weight, kernel_constraint=constraints, kernel_initializer=kernel_initializer)(x)
        if bn:
            x = self.bnFunc()(x)
        x = self.activations()(x)
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
        return x
    
    def _upsample_block_transp(self, x, filters, kernel_size, dropout_rate=None, bn=False, normalize_weight=False, constraints=None, kernel_initializer='complex_independent'):
        if self.complex_network:
            x = self.convFunc(filters, kernel_size, strides=2, padding='same', use_bias=self.use_bias, transposed=True, normalize_weight=normalize_weight, kernel_constraint=constraints, kernel_initializer=kernel_initializer)(x)
        else:
            x = Conv2DTranspose(filters, kernel_size, strides=2, padding='same', use_bias=self.use_bias, normalize_weight=normalize_weight, kernel_constraint=constraints)(x)
        x = self.activations()(x)
        x = self.convFunc(filters, kernel_size, padding='same', use_bias=self.use_bias, normalize_weight=normalize_weight, kernel_constraint=constraints)(x)
        # x = self.activations()(x)
        # x = self.convFunc(filters, kernel_size, padding='same', use_bias=self.use_bias, normalize_weight=normalize_weight, kernel_constraint=constraints)(x)
        if bn:
            x = self.bnFunc()(x)
        x = self.activations()(x)
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
        return x
    
    def build_model(self):
        inputs = Input(self.input_shape)
        # constraints = tf.keras.constraints.MinMaxNorm(1., 10.)
        constraints=None
        if self.forward:
            inputs_forward = Input(self.input_shape)            
        x = inputs
        # downsample
        skips = []
        kernel_initializer = 'complex'
        normalize_weight = False
        filters_factor = [4,8,16,32,64]
        # dropout_rates = [None,None,0.4,0.6,0.8]
        dropout_rates = [None,None,0.3,0.5,0.7]
        for ithlayer in range(self.num_downsample_layer):
            if ithlayer <= 0:
                x = self._downsample_block_conv(x, filters_factor[ithlayer]*self.filters, self.size, normalize_weight=normalize_weight, constraints=constraints, kernel_initializer=kernel_initializer) # downsampling 2^num_downsample_layer
            else:
                x = self._downsample_block_conv(x, filters_factor[ithlayer]*self.filters, self.size, dropout_rate=dropout_rates[ithlayer],bn=True, normalize_weight=normalize_weight, constraints=constraints, kernel_initializer=kernel_initializer) # downsampling 2^num_downsample_layer
            skips.append(x)
        x = self.convFunc(filters_factor[ithlayer]*self.filters, self.size, use_bias=self.use_bias, padding='same', normalize_weight=normalize_weight, kernel_constraint=constraints, kernel_initializer=kernel_initializer)(x)
        x = self.bnFunc()(x)
        x = self.activations()(x)
        # upsample
        skips = reversed(skips)
        for skip in skips:
            x = self.concatFunc()([x,skip])
            if ithlayer <= 0:
                x = self._upsample_block_transp(x, filters_factor[ithlayer]*self.filters, self.size,  normalize_weight=normalize_weight, constraints=constraints, kernel_initializer=kernel_initializer)
            else:
                x = self._upsample_block_transp(x, filters_factor[ithlayer]*self.filters, self.size,  dropout_rate=dropout_rates[ithlayer], bn=True, normalize_weight=normalize_weight, constraints=constraints, kernel_initializer=kernel_initializer)
            ithlayer = ithlayer - 1
        # for ii in range(1,4):
        #     x = self.convFunc(2**ii*self.filters, 3, padding='same', use_bias=self.use_bias, normalize_weight=normalize_weight, kernel_constraint=constraints, kernel_initializer=kernel_initializer)(x)
        #     x = ComplexNormalization()(x)
        #     x = self.activations()(x)
        x = self.convFunc(1, (3,3), padding='same', use_bias=self.use_bias, normalize_weight=normalize_weight, kernel_constraint=constraints, kernel_initializer=kernel_initializer)(x)
        if self.forward:
            x = self.activations()(x)
            x = Concatenate(axis=-1)((x, inputs_forward))
            x = self.convFunc(2, 3, padding='same', use_bias=self.use_bias)(x)
            x = tf.keras.layers.LeakyReLU()(x)
            return tf.keras.Model(inputs=[inputs, inputs_forward], outputs=x)
        else:    
            x = self.activations()(x)
            return tf.keras.Model(inputs=inputs, outputs=x)
        
class TestUNet1(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build_model(self):
        inputs = Input(self.input_shape)
        # constraints = tf.keras.constraints.MinMaxNorm(1., 10.)
        constraints=None
        if self.forward:
            inputs_forward = Input(self.input_shape)            
        x = inputs
        # downsample
        skips = []
        kernel_initializer = 'complex'
        normalize_weight = False
        for ithlayer in range(self.num_downsample_layer):
            if ithlayer <= 2:
                x = self.downsample_block_conv(x, 2**(ithlayer+1)*self.filters, self.size, normalize_weight=normalize_weight, constraints=constraints, kernel_initializer=kernel_initializer) # downsampling 2^num_downsample_layer
                skips.append(x)
            else:
                x = self.downsample_block_conv(x, 2**(ithlayer+1)*self.filters, self.size, bn=True, normalize_weight=normalize_weight, constraints=constraints, kernel_initializer=kernel_initializer) # downsampling 2^num_downsample_layer
        x = self.convFunc(2**(ithlayer+1)*self.filters, self.size, padding='same', normalize_weight=normalize_weight, kernel_constraint=constraints, kernel_initializer=kernel_initializer)(x)
        x = self.bnFunc()(x)
        x = self.activations()(x)
        # upsample
        skips = reversed(skips)
        for ithlayer in range(self.num_downsample_layer,3,-1):
            x = self.upsample_block_PS(x, 2**(ithlayer+1)*self.filters, self.size, bn=True, normalize_weight=normalize_weight, constraints=constraints, kernel_initializer=kernel_initializer)
        for skip in skips:
            x = self.concatFunc()([x,skip])
            x = self.upsample_block_transp(x, 2**(ithlayer+1)*self.filters, self.size,  normalize_weight=normalize_weight, constraints=constraints, kernel_initializer=kernel_initializer)
            ithlayer = ithlayer - 1
        for _ in range(3):
            x = self.convFunc(2*self.filters, 3, padding='same', use_bias=self.use_bias, normalize_weight=normalize_weight, kernel_constraint=constraints, kernel_initializer=kernel_initializer)(x)
            x = self.activations()(x)
        x = self.convFunc(1, (1,1), padding='same', use_bias=self.use_bias, normalize_weight=normalize_weight, kernel_constraint=constraints, kernel_initializer=kernel_initializer)(x)
        if self.forward:
            x = self.activations()(x)
            x = Concatenate(axis=-1)((x, inputs_forward))
            x = self.convFunc(2, 3, padding='same', use_bias=self.use_bias)(x)
            x = tf.keras.layers.LeakyReLU()(x)
            return tf.keras.Model(inputs=[inputs, inputs_forward], outputs=x)
        else:    
            x = self.activations()(x)
            return tf.keras.Model(inputs=inputs, outputs=x)
        
class AMUNet(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _downsample_block_conv(self, x, filters, kernel_size, dropout_rate=None, bn=False, normalize_weight=False, constraints=None, kernel_initializer='complex_independent'):      
        x = self.convFunc(filters, kernel_size, strides=2, padding='same', use_bias=self.use_bias, normalize_weight=normalize_weight, kernel_constraint=constraints, kernel_initializer=kernel_initializer)(x)
        if bn:
            x = self.bnFunc()(x)
        x = self.activations()(x)
        x = self.convFunc(filters, kernel_size, padding='same', use_bias=self.use_bias, normalize_weight=normalize_weight, kernel_constraint=constraints, kernel_initializer=kernel_initializer)(x)
        if bn:
            x = self.bnFunc()(x)
        x = self.activations()(x)
        x = self.convFunc(filters, kernel_size, padding='same', use_bias=self.use_bias, normalize_weight=normalize_weight, kernel_constraint=constraints, kernel_initializer=kernel_initializer)(x)
        if bn:
            x = self.bnFunc()(x)
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
        return x
    
    def _upsample_block_interp(self, x, filters, kernel_size, dropout_rate=None, bn=False, normalize_weight=False, constraints=None, kernel_initializer='complex_independent'):
        x = tf.keras.layers.UpSampling2D((2,2), interpolation='bilinear')(x)
        if bn:
            x = self.bnFunc()(x)
        x = self.activations()(x)
        x = self.convFunc(filters, kernel_size, padding='same', use_bias=self.use_bias, normalize_weight=normalize_weight, kernel_constraint=constraints, kernel_initializer=kernel_initializer)(x)
        if bn:
            x = self.bnFunc()(x)
        x = self.activations()(x)
        x = self.convFunc(filters, kernel_size, padding='same', use_bias=self.use_bias, normalize_weight=normalize_weight, kernel_constraint=constraints, kernel_initializer=kernel_initializer)(x)
        if bn:
            x = self.bnFunc()(x)
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
        return x
    
    def build_model(self):
        inputs = Input(self.input_shape)
        constraints = tf.keras.constraints.UnitNorm()
        # constraints=None
        if self.forward:
            inputs_forward = Input(self.input_shape)            
        x = inputs
        # downsample
        kernel_initializer = 'complex'
        normalize_weight = False
        skips = []
        for ithlayer in range(self.num_downsample_layer):
            if ithlayer > 1:
                bn = True
                x = self._downsample_block_conv(x, 2**(ithlayer+1)*self.filters, self.size, bn=True, normalize_weight=normalize_weight, constraints=constraints, kernel_initializer=kernel_initializer) # downsampling 2^num_downsample_layer
            else:
                bn = False
                x = self._downsample_block_conv(x, 2**(ithlayer+1)*self.filters, self.size, bn=bn, normalize_weight=normalize_weight, constraints=constraints, kernel_initializer=kernel_initializer) # downsampling 2^num_downsample_layer
            skips.append(x)
        x = self.convFunc(2**(ithlayer+1)*self.filters, self.size, padding='same', normalize_weight=normalize_weight, kernel_constraint=constraints, kernel_initializer=kernel_initializer)(x)
        x = self.bnFunc()(x)
        # upsample
        for ithlayer in range(self.num_downsample_layer-1,-1,-1):
            if ithlayer > 1:
                bn = True
                x = self.concatFunc()([x, skips[-1]])
                x = self.upsample_block_transp(x, 2**(ithlayer+1)*self.filters, self.size, bn=bn, normalize_weight=normalize_weight, constraints=constraints, kernel_initializer=kernel_initializer)
                skips.pop(-1)
            else:
                bn = False
                x = self.upsample_block_PS(x, 2**(ithlayer+1)*self.filters, self.size, bn=bn, normalize_weight=normalize_weight, constraints=constraints, kernel_initializer=kernel_initializer)
        for _ in range(3):
            x = self.convFunc(2*self.filters, 3, padding='same', use_bias=self.use_bias, normalize_weight=normalize_weight, kernel_constraint=constraints, kernel_initializer=kernel_initializer)(x)
            x = self.activations()(x)
        x = self.convFunc(1, (1,1), padding='same', use_bias=self.use_bias, normalize_weight=normalize_weight, kernel_constraint=constraints, kernel_initializer=kernel_initializer)(x)
        if self.forward:
            x = self.activations()(x)
            x = Concatenate(axis=-1)((x, inputs_forward))
            x = self.convFunc(2, 3, padding='same', use_bias=self.use_bias)(x)
            x = tf.keras.layers.LeakyReLU()(x)
            return tf.keras.Model(inputs=[inputs, inputs_forward], outputs=x)
        else:    
            x = ComplexNormalization()(x)
            return tf.keras.Model(inputs=inputs, outputs=x)
    
class UNet3plus(BaseModel):
    # ref: https://github.com/hamidriasat/UNet-3-Plus/blob/unet3p_lits/models/unet3plus.py
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def conv_block(self, x, kernels, kernel_size=(3, 3), strides=(1, 1), padding='same',
               is_bn=True, is_relu=True, n=2):
        for i in range(1, n + 1):
            x = self.convFunc(filters=kernels,
                              kernel_size=kernel_size,
                              padding=padding,
                              strides=strides,
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        if is_bn:
            x = self.bnFunc()(x)
        if is_relu:
            x = self.activations()(x)

        return x
    
    def build_model(self):
        """ UNet3+ base model """
        #filters = [64, 128, 256, 512, 1024]
        filters = [self.filters*2**ii for ii in range(1,6)]
         
        input_layer = tf.keras.layers.Input(shape=self.input_shape,name="input_layer")  # 320*320*3
         
        """ Encoder"""
        # block 1
        e1 = self.conv_block(input_layer, filters[0])  # 320*320*64
         
        # block 2
        e2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(e1)  # 160*160*64
        e2 = self.conv_block(e2, filters[1])  # 160*160*128
         
        # block 3
        e3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(e2)  # 80*80*128
        e3 = self.conv_block(e3, filters[2])  # 80*80*256
         
        # block 4
        e4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(e3)  # 40*40*256
        e4 = self.conv_block(e4, filters[3])  # 40*40*512
         
        # block 5
        # bottleneck layer
        e5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(e4)  # 20*20*512
        e5 = self.conv_block(e5, filters[4])  # 20*20*1024
         
        """ Decoder """
        cat_channels = filters[0]
        cat_blocks = len(filters)
        upsample_channels = cat_blocks * cat_channels
         
        """ d4 """
        e1_d4 = tf.keras.layers.MaxPool2D(pool_size=(8, 8))(e1)  # 320*320*64  --> 40*40*64
        e1_d4 = self.conv_block(e1_d4, cat_channels, n=1)  # 320*320*64  --> 40*40*64
         
        e2_d4 = tf.keras.layers.MaxPool2D(pool_size=(4, 4))(e2)  # 160*160*128 --> 40*40*128
        e2_d4 = self.conv_block(e2_d4, cat_channels, n=1)  # 160*160*128 --> 40*40*64
         
        e3_d4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(e3)  # 80*80*256  --> 40*40*256
        e3_d4 = self.conv_block(e3_d4, cat_channels, n=1)  # 80*80*256  --> 40*40*64
         
        e4_d4 = self.conv_block(e4, cat_channels, n=1)  # 40*40*512  --> 40*40*64
         
        e5_d4 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(e5)  # 80*80*256  --> 40*40*256
        e5_d4 = self.conv_block(e5_d4, cat_channels, n=1)  # 20*20*1024  --> 20*20*64
         
        d4 = tf.keras.layers.concatenate([e1_d4, e2_d4, e3_d4, e4_d4, e5_d4])
        d4 = self.conv_block(d4, upsample_channels, n=1)  # 40*40*320  --> 40*40*320
         
        """ d3 """
        e1_d3 = tf.keras.layers.MaxPool2D(pool_size=(4, 4))(e1)  # 320*320*64 --> 80*80*64
        e1_d3 = self.conv_block(e1_d3, cat_channels, n=1)  # 80*80*64 --> 80*80*64
         
        e2_d3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(e2)  # 160*160*256 --> 80*80*256
        e2_d3 = self.conv_block(e2_d3, cat_channels, n=1)  # 80*80*256 --> 80*80*64
         
        e3_d3 =self. conv_block(e3, cat_channels, n=1)  # 80*80*512 --> 80*80*64
         
        e4_d3 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d4)  # 40*40*320 --> 80*80*320
        e4_d3 = self.conv_block(e4_d3, cat_channels, n=1)  # 80*80*320 --> 80*80*64
         
        e5_d3 = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(e5)  # 20*20*320 --> 80*80*320
        e5_d3 = self.conv_block(e5_d3, cat_channels, n=1)  # 80*80*320 --> 80*80*64
         
        d3 = tf.keras.layers.concatenate([e1_d3, e2_d3, e3_d3, e4_d3, e5_d3])
        d3 = self.conv_block(d3, upsample_channels, n=1)  # 80*80*320 --> 80*80*320
         
        """ d2 """
        e1_d2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(e1)  # 320*320*64 --> 160*160*64
        e1_d2 = self.conv_block(e1_d2, cat_channels, n=1)  # 160*160*64 --> 160*160*64
         
        e2_d2 = self.conv_block(e2, cat_channels, n=1)  # 160*160*256 --> 160*160*64
         
        d3_d2 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d3)  # 80*80*320 --> 160*160*320
        d3_d2 = self.conv_block(d3_d2, cat_channels, n=1)  # 160*160*320 --> 160*160*64
         
        d4_d2 = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(d4)  # 40*40*320 --> 160*160*320
        d4_d2 = self.conv_block(d4_d2, cat_channels, n=1)  # 160*160*320 --> 160*160*64
         
        e5_d2 = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(e5)  # 20*20*320 --> 160*160*320
        e5_d2 = self.conv_block(e5_d2, cat_channels, n=1)  # 160*160*320 --> 160*160*64
         
        d2 = tf.keras.layers.concatenate([e1_d2, e2_d2, d3_d2, d4_d2, e5_d2])
        d2 = self.conv_block(d2, upsample_channels, n=1)  # 160*160*320 --> 160*160*320
         
        """ d1 """
        e1_d1 = self.conv_block(e1, cat_channels, n=1)  # 320*320*64 --> 320*320*64
         
        d2_d1 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d2)  # 160*160*320 --> 320*320*320
        d2_d1 = self.conv_block(d2_d1, cat_channels, n=1)  # 160*160*320 --> 160*160*64
         
        d3_d1 = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(d3)  # 80*80*320 --> 320*320*320
        d3_d1 = self.conv_block(d3_d1, cat_channels, n=1)  # 320*320*320 --> 320*320*64
         
        d4_d1 = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(d4)  # 40*40*320 --> 320*320*320
        d4_d1 = self.conv_block(d4_d1, cat_channels, n=1)  # 320*320*320 --> 320*320*64
         
        e5_d1 = tf.keras.layers.UpSampling2D(size=(16, 16), interpolation='bilinear')(e5)  # 20*20*320 --> 320*320*320
        e5_d1 = self.conv_block(e5_d1, cat_channels, n=1)  # 320*320*320 --> 320*320*64
         
        d1 = tf.keras.layers.concatenate([e1_d1, d2_d1, d3_d1, d4_d1, e5_d1, ])
        d1 = self.conv_block(d1, upsample_channels, n=1)  # 320*320*320 --> 320*320*320
         
        # last layer does not have batchnorm and relu
        d = self.conv_block(d1, 1, n=1, is_bn=False, is_relu=False)
         
        output = self.activations()(d)
         
        return tf.keras.Model(inputs=input_layer, outputs=output, name='UNet_3Plus')

class mobilenetv2(BaseModel):
    # reference:https://github.com/monatis/mobilenetv2-tf2/blob/master/mobilenetv2.py
    def __init__(self,
                 batch_size=32,
                 lr=1e-3,
                 epochs=500,
                 seed=7414,
                 losses='MSE',
                 complex_network=False,
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         lr=lr,
                         epochs=epochs,
                         seed=seed,
                         losses=losses,
                         complex_network=complex_network,
                         **kwargs)
    
    def relu6(self):
        return tf.keras.layers.ReLU(6.)
    def _conv_block(self, inputs, filters, kernel, strides):
        """Convolution Block
        This function defines a 2D convolution operation with BN and relu6.
    
        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution along the width and height.
                Can be a single integer to specify the same value for
                all spatial dimensions.
    
        # Returns
            Output tensor.
        """
        x = self.convFunc(filters, kernel, padding='same', strides=strides)(inputs)
        x = self.bnFunc()(x)
        return self.relu6()(x)
    
    def _bottleneck(self, inputs, filters, kernel, t, s, r=False):
        """Bottleneck
        This function defines a basic bottleneck structure.
    
        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            t: Integer, expansion factor.
                t is always applied to the input size.
            s: An integer or tuple/list of 2 integers,specifying the strides
                of the convolution along the width and height.Can be a single
                integer to specify the same value for all spatial dimensions.
            r: Boolean, Whether to use the residuals.
    
        # Returns
            Output tensor.
        """
    
        tchannel = inputs.shape[-1] * t
    
        x = self._conv_block(inputs, tchannel, (1, 1), (1, 1))
    
        x = tf.keras.layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
        x = self.bnFunc()(x)
        x = self.relu6()(x)
    
        x = self.convFunc(filters, (1, 1), strides=(1, 1), padding='same')(x)
        x = self.bnFunc()(x)
    
        if r:
            x = tf.keras.layers.add([x, inputs])
        return x
    
    def _inverted_residual_block(self, inputs, filters, kernel, t, strides, n):
        """Inverted Residual Block
        This function defines a sequence of 1 or more identical layers.
    
        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            t: Integer, expansion factor.
                t is always applied to the input size.
            s: An integer or tuple/list of 2 integers,specifying the strides
                of the convolution along the width and height.Can be a single
                integer to specify the same value for all spatial dimensions.
            n: Integer, layer repeat times.
        # Returns
            Output tensor.
        """
    
        x = self._bottleneck(inputs, filters, kernel, t, strides)
    
        for i in range(1, n):
            x = self._bottleneck(x, filters, kernel, t, 1, True)
    
        return x

    def build_model(self):
        """MobileNetv2
        This function defines a MobileNetv2 architecture.
    
        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor.
            k: Integer, number of classes.
            plot_model: Boolean, whether to plot model architecture or not
        # Returns
            MobileNetv2 model.
        """
        inputs = tf.keras.layers.Input(shape=self.input_shape, name='input')
        x = self._conv_block(inputs, 32, (3, 3), strides=(2, 2))
    
        x = self._inverted_residual_block(x, 16, (3, 3), t=1, strides=1, n=1)
        x = self._inverted_residual_block(x, 24, (3, 3), t=6, strides=2, n=2)
        x = self._inverted_residual_block(x, 32, (3, 3), t=6, strides=2, n=3)
        x = self._inverted_residual_block(x, 64, (3, 3), t=6, strides=2, n=4)
        x = self._inverted_residual_block(x, 96, (3, 3), t=6, strides=1, n=3)
        x = self._inverted_residual_block(x, 160, (3, 3), t=6, strides=2, n=3)
        x = self._inverted_residual_block(x, 320, (3, 3), t=6, strides=1, n=1)
    
        x = self._conv_block(x, 1280, (1, 1), strides=(1, 1))
        # x = tf.keras.layers.GlobalAveragePooling2D()(x)
        # x = tf.keras.layers.Reshape((1, 1, 1280))(x)
        # x = tf.keras.layers.Dropout(0.8, name='Dropout')(x)
        # x = tf.keras.layers.Conv2D(128, (1, 1), padding='same')(x)
        # x = tf.keras.layers.Activation('softmax', name='final_activation')(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        output = tf.keras.layers.Dense(128)(x)
        return tf.keras.models.Model(inputs=inputs, outputs=output)

class CVCNN(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _conv1D(self, filters, kernel, strides, padding, transposed):
        if transposed:
            return tf.keras.layers.Conv1DTranspose(filters, kernel, strides, padding)
        else:
            return tf.keras.layers.Conv1D(filters, kernel, strides, padding)
    def _conv_block1(self, inputs):
        x = self.convFunc(256,(3,3), strides=2)(inputs)
        x = self.bnFunc()(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.convFunc(128, (2,3), strides=2)(x)
        x = self.bnFunc()(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.convFunc(32, (1,1), strides=2)(x)
        x = self.bnFunc()(x)
        x = tf.keras.layers.ReLU()(x)
        x = Dropout(0.5)(x)
        return x
    
    def _conv_block2(self, inputs):
        x = self.convFunc(64,(3,3), strides=2)(inputs)
        x = self.bnFunc()(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.convFunc(32, (4,4), strides=2)(x)
        x = self.bnFunc()(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.convFunc(32, (1,1), strides=2)(x)
        x = self.bnFunc()(x)
        x = tf.keras.layers.ReLU()(x)
        x = Dropout(0.5)(x)
        x = self.convFunc(16, (3,3), strides=2)(x)
        x = self.bnFunc()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        return x
    
    def _inception(self, inputs):
        # block 1
        x1 = self.convFunc(48, (1,1), padding='same')(inputs)
        x1 = self.bnFunc()(x1)
        x1 = tf.keras.layers.ReLU()(x1)
        x1 = self.convFunc(64, (5,5), padding='same')(x1)
        x1 = self.bnFunc()(x1)
        x1 = tf.keras.layers.ReLU()(x1)
        # block 2
        x2 = self.convFunc(64, (1,1), padding='same')(inputs)
        x2 = self.bnFunc()(x2)
        x2 = tf.keras.layers.ReLU()(x2)
        x2 = self.convFunc(96, (3,3), padding='same')(x2)
        x2 = self.bnFunc()(x2)
        x2 = tf.keras.layers.ReLU()(x2)
        x2 = self.convFunc(96, (3,3), padding='same')(x2)
        x2 = self.bnFunc()(x2)
        x2 = tf.keras.layers.ReLU()(x2)

        # block 3
        x3 = self.convFunc(64, (1,1), padding='same')(inputs)
        x3 = self.bnFunc()(x3)
        x3 = tf.keras.layers.ReLU()(x3)
        # block 4
        x4 = self.convFunc(32, (3,3), padding='same')(inputs)
        x4 = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(x4)
        x4 = self.bnFunc()(x4)
        x4 = tf.keras.layers.ReLU()(x4)
        x = self.concatFunc()([x1,x2,x3,x4])
        x = Dropout(0.5)(x)
        return x
    
    def Vnet(self, inputs):
        x = tf.keras.layers.Reshape((inputs.shape[1],1))(inputs)
        x = self._conv1D(16, 3, 2, 'same', False)(x)
        x = self.bnFunc()(x)
        x = tf.keras.layers.ReLU()(x)
        skip1 = x
        x = self._conv1D(32, 3, 2, 'same', False)(x)
        x = self.bnFunc()(x)
        x = tf.keras.layers.ReLU()(x)
        x = self._conv1D(32, 3, 1, 'same', False)(x)
        x = self.bnFunc()(x)
        x = tf.keras.layers.ReLU()(x)
        skip2 = x
        x = self._conv1D(64, 3, 2, 'same', False)(x)
        x = self.bnFunc()(x)
        x = tf.keras.layers.ReLU()(x)
        x = self._conv1D(64, 3, 1, 'same', False)(x)
        x = self.bnFunc()(x)
        x = tf.keras.layers.ReLU()(x)
        # upsample
        x = self._conv1D(32, 3, 2, 'same', True)(x)
        x = self.bnFunc()(x)
        x = tf.keras.layers.ReLU()(x)
        x = self._conv1D(32, 3, 1, 'same', False)(x)
        x = self.bnFunc()(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.concatFunc()([x,skip2])
        x = self._conv1D(16, 3, 2, 'same', True)(x)
        x = self.bnFunc()(x)
        x = tf.keras.layers.ReLU()(x)
        x = self._conv1D(16, 3, 1, 'same', False)(x)
        x = self.bnFunc()(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.concatFunc()([x,skip1])
        # Deconv
        x = self._conv1D(8, 3, 1, 'same', False)(x)
        x = self.bnFunc()(x)
        x = tf.keras.layers.ReLU()(x)
        x = self._conv1D(8, 3, 2, 'same', True)(x)
        x = self.bnFunc()(x)
        x = tf.keras.layers.ReLU()(x)
        x = self._conv1D(2, 3, 1, 'same', False)(x)
        x = self.bnFunc()(x)
        x = tf.keras.layers.ReLU()(x)
        x = self._conv1D(2, 3, 2, 'same', True)(x)
        x = self.bnFunc()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128)(x)
        return x
    
    def build_model(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape, name='input')
        x = self._conv_block1(inputs)
        x = self._inception(x)
        x = self._conv_block2(x)
        output = self.Vnet(x)
        return tf.keras.models.Model(inputs=inputs, outputs=output)
        

    
        
def getModel(model_type):
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
    elif model_type in {'RevisedSegNet', 'revisedsegnet', 'revisedSegnet'}:
        return RevisedSegNet
    elif model_type in {'srn', 'SRN'}:
        return SRN
    elif model_type in {'test', 'testUNet', 'testunet', 'TestUNet'}:
        return TestUNet
    elif model_type in {'resunet', 'ResUNet', 'ResUnet'}:
        return ResUNet
    elif model_type in {'amuet', 'AMUnet', 'AMUNet'}:
        return AMUNet
    elif model_type in {'UNet3plus','unet3+','unet3p','unet3plus'}:
        return UNet3plus
    elif model_type in {'mobilenet','mobileNet','MobileNet'}:
        return  mobilenetv2
    elif model_type in {'cvcnn','CVCNN'}:
        return CVCNN