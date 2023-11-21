# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 17:19:21 2023

@author: benzener
"""
import tensorflow as tf
import os
import numpy as np
import time
from complexnn.activation import cReLU, zReLU, modReLU, AmplitudeMaxout, FLeakyReLU, ctanh, complexReLU, mish, cLeakyReLU
from complexnn.loss import ComplexRMS, ComplexMAE, ComplexMSE, SSIM, MS_SSIM, SSIM_MSE, aSSIM_MSE, ssim_map, MSE, MAE, wMSE
from complexnn.conv_test import ComplexConv2D, ComplexConv1D
from complexnn.bn_test import ComplexBatchNormalization
from complexnn.pool_test import ComplexMaxPoolingWithArgmax2D, MaxUnpooling2D
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
    from baseband.utils.fig_utils import Bmode_fig
    from baseband.utils.data_utils import normalization, convert_to_real
    from baseband.model_utils import InOne, CosineDecay, ComplexNormalization, ComplexConcatenate, GlobalAveragePooling
    sys.path.remove(addpath)
else:
    from ..setting import constant
    from ..utils.fig_utils import Bmode_fig
    from ..utils.data_utils import normalization, convert_to_real
    from .model_utils import InOne, CosineDecay, ComplexNormalization, ComplexConcatenate, GlobalAveragePooling


class BaseModel(): 
    '''
    Based model architecture.
    Args:
        filter: int, number of filters in the first layer.
        size: int or tuple, filter size for each layer.
        batch_size: int, mini-batch size.
        lr: scalar, learning rate.
        epochs: int, training epochs.
        validation_rate: float, percentage of validation in the whole input training dataset.
        validation_data: ndarray, validation dataset. If `validation_rate` and `validation_data` concurrently exists,
            use `validation_data` and ignore `validation_rate`.
        seed: int, random seed.
        activations: string, acitvation function.
        losses: string, loss function.
        dropout: boolean, whether to use dropout layer.
        use_bias: boolean, whether to use bias in layers.
        complex_network: boolean, RF (False) or BB (True) model.
        batchsizeschedule:
    '''
    def __init__(self,
                 filters=16,
                 size=(3,3),
                 batch_size=32,
                 lr=1e-3,
                 epochs=1000,
                 validation_rate=None,
                 validation_data=None,
                 num_downsample_layer=None,
                 seed=7414,
                 activations='LeakyReLU',
                 losses='SSIM',
                 optimizer='Adam',
                 dropout=False,
                 use_bias=False,
                 complex_network=True,
                 batchsizeschedule=False):
        self.filters = filters
        self.size = size
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.validation_rate = validation_rate
        self.validation_data = validation_data
        self.num_downsample_layer = num_downsample_layer
        self.seed = seed
        self.activations = activations
        self.losses = losses
        self.optimizer = optimizer
        self.dropout = dropout
        self.use_bias = use_bias
        self.complex_network = complex_network
        self.batchsizeschedule = batchsizeschedule

        
        

    def bn_act_drop_block(self, x, bn, act, dropout_rate):
        if bn:
            x = self.bnFunc()(x)
        if act is not None:
            try:
                x = act()(x)
            except TypeError:
                x = act(x)
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
        return x

    
    def conv_block(self, 
                   x,
                   filters,
                   kernel_size,
                   bn=False, 
                   act=None,
                   use_bias=True,
                   dropout_rate=None,
                   strides=1,
                   padding='same',
                   transposed=False,
                   normalize_weight=False,
                   kernel_initializer=None,
                   bias_initializer="zeros",
                   activity_regularizer=None,
                   kernel_regularizer=None,
                   kernel_constraints=None,
                   bias_constraint=None,
                   ):
        '''
        Args:
            filters: Integer, the dimensionality of the complex output space (i.e, the number complex feature maps in the convolution). The
                total effective number of filters or feature maps is 2 x filters.
            kernel_size: An integer or tuple/list of 2 integers, specifying the width and height of the 2D convolution windw. 
                Can be a single integer to specify the same value for all spatial dimensions.
            bn: Boolean, whether to use batch normalization.
            act: Activation function to use.
            strides: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the width and height.
                Can be a single integer to specify the same value for all spatial dimensions. 
                Specifying any stride value != 1 is incompatible with specifying any `dilation_rate` value != 1.
            padding: one of `"valid"` or `"same"` (case-insensitive).
            transposed: Boolean, whether or not to use transposed convolution
            
            normalize_weight: Boolean, whether the layer normalizes its complex weights before convolving the complex input.
                The complex normalization performed is similar to the one for the batchnorm. Each of the complex kernels are 
                centred and multiplied by the inverse square root of covariance matrix. Then, a complex multiplication is perfromed as 
                the normalized weights are multiplied by the complex scaling factor gamma.
            kernel_initializer: Initializer for the weights matrix.
                By default it is 'complex' for complex model or `glorot_uniform` for real model. 
                For complex model, 'complex_independent' and the usual initializers could also be used. (see keras.initializers and init.py).
            bias_initializer: Initializer for the bias vector (see keras.initializers).
            kernel_regularizer: Regularizer function applied to the `kernel` weights matrix (see keras.regularizers).
            bias_regularizer: Regularizer function applied to the bias vector (see keras.regularizers).
            activity_regularizer: Regularizer function applied to the output of the layer (its "activation"). (see keras.regularizers).
            kernel_constraint: Constraint function applied to the kernel matrix (see keras.constraints).
            bias_constraint: Constraint function applied to the bias vector (see keras.constraints).
            
        '''
        kwargs = {
            'filters':filters,
            'kernel_size':kernel_size,
            'use_bias':use_bias,
            'strides':strides,
            'padding':padding,
            'kernel_initializer':kernel_initializer if kernel_initializer is not None else 'complex' if self.complex_network else 'glorot_uniform',
            'bias_initializer':bias_initializer,
            'activity_regularizer':activity_regularizer,
            'kernel_regularizer':kernel_regularizer,
            'kernel_constraint':kernel_constraints,
            'bias_constraint':bias_constraint
            }
        if self.complex_network:
            convFunc = ComplexConv2D(**kwargs, transposed=transposed,normalize_weight=normalize_weight)
        else:
            if transposed:
                convFunc = Conv2DTranspose(**kwargs)
            else:
                convFunc = Conv2D(**kwargs)
        x = convFunc(x)
        return self.bn_act_drop_block(x, bn, act, dropout_rate)

    
    def downsample_block_argmax(self, x, bn=False, act=None, dropout_rate=None, **kwargs):
        # downsampling using maxpool and returns indices
        downsampleFunc = ComplexMaxPoolingWithArgmax2D((2,2), 2, **kwargs) if self.complex_network else tf.keras.layers.MaxPool2D((2,2), 2, 'same', **kwargs)
        x, inds = downsampleFunc(x) # return list [x, indics]
        return self.bn_act_drop_block(x, bn, act, dropout_rate), inds
    
    def downsample_block_maxpool(self, x, bn=False, act=None, dropout_rate=None, **kwargs):
        # downsampling using maxpool
        x, _ = self.downsample_block_argmax(x, bn=False, act=None, dropout_rate=None, **kwargs)
        return x
    
    def upsample_block_PS(self, x, bn=False, act=None, dropout_rate=None):
        # upsampling using pixel shuffle
        x = PixelShuffler((2,2))(x)
        return self.bn_act_drop_block(x, bn, act, dropout_rate)
    
    def upsample_block_unpool(self, xandind, bn=False, act=None, dropout_rate=None):
        # upsampling using index pool. input is a list [x, indices]
        x = MaxUnpooling2D((2,2))(xandind)
        return self.bn_act_drop_block(x, bn, act, dropout_rate)
    
    def upsample_block_interp(self, x, bn=False, act=None, dropout_rate=None, interpolation='bilinear'):
        # upsampling using 2D interpolation
        x = tf.keras.layers.UpSampling2D((2,2), interpolation=interpolation)(x)
        return self.bn_act_drop_block(x, bn, act, dropout_rate)
        
    
    def identity_block(self, x, filters, kernel_size, bn=False, dropout_rate=None):
        last_layer_filters = x.shape[-1]//2 if self.complex_network else x.shape[-1]
        shortcut = x
        x = self.convFunc(filters, (1,1), padding='same', strides=1, use_bias=self.use_bias)(x)
        if bn:
            x = self.bnFunc()(x)
        x = self.activations()(x)
        x = self.convFunc(filters, kernel_size, padding='same', strides=1, use_bias=self.use_bias)(x)
        if bn:
            x = self.bnFunc()(x)
        x = self.activations()(x)
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
        x = self.convFunc(last_layer_filters, (1,1), padding='same', strides=1, use_bias=self.use_bias)(x)
        if bn:
            x = self.bnFunc()(x)
        x = self.activations()(x)
        x = Add()([x, shortcut])
        x = self.activations()(x)
        return x
                    
    def get_custom_object(self):
        # if error happens when loading model and get unknown custom object, you need to add the `unknown` function
        # into this dictionary to let keras know your custom object.
        custom_object = {
            'ComplexConv2D':ComplexConv2D,
            'ComplexBatchNormalization':ComplexBatchNormalization,
            'MaxPoolingWithArgmax2D':ComplexMaxPoolingWithArgmax2D,
            'MaxUnpooling2D':MaxUnpooling2D,
            'ComplexMSE':ComplexMSE,
            'MSE':MSE,
            'MAE':MAE,
            'ctanh':ctanh,
            'modReLU':modReLU,
            'FLeakyReLU': FLeakyReLU,
            'AmplitudeMaxout':AmplitudeMaxout,
            'complexReLU':complexReLU,
            'mish':mish,
            'wMSE':wMSE
            }
        return custom_object
    
    def _get_name(self, obj):
        if isinstance(obj, str):
            return obj
        try:
            name = obj.__name__
        except AttributeError:
            name = obj.__class__.__name__
        return name
        
    
    def _generate_modelname(self):
        now = datetime.now()
        day_month_year = now.strftime("%d%m%Y")
        data_type = 'complex' if self.complex_network else 'real'
        model_name = self.__class__.__name__
        epochs = str(self.epochs)
        loss_name = self._get_name(self.losses)
        act_name = self._get_name(self.activations)
        opt_name = self._get_name(self.optimizer)
        if self.fine_tune:
            self.model_name = f'{data_type}{model_name}_{epochs}_{loss_name}_{act_name}_{opt_name}_finetune_{day_month_year}'
        else:
            self.model_name = f'{data_type}{model_name}_{epochs}_{loss_name}_{act_name}_{opt_name}_{day_month_year}'
    
    def _build_info(self):
        self.model_info = {
            'input_shape':self.input_shape,
            'complex':self.complex_network,
            'validation_split':self.validation_rate,
            'filters':self.filters,
            'kernel_size':self.size,
            'learning_rate':self.lr,
            'batch_size':self.batch_size,
            'epochs':self.epochs,
            'activation':self._get_name(self.activations),
            'loss':self._get_name(self.losses),
            'optimizer':self._get_name(self.optimizer),
            'dropout':self.dropout,
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
        # convert "string" to func
        default_acts = {
            True:{'crelu', 'zrelu', 'modrelu','cleakyrelu', 'ctanh','leakyrelu', 'amu', 'fleakyrelu', 'complexrelu', 'mish'},
            False:{'leakyrelu','flearkyrelu','relu'}
            }[self.complex_network]
        if isinstance(self.activations, str):
            self.activations = self.activations.lower()
            if self.activations in default_acts:
                self.activations = {
                    'crelu': cReLU,
                    'zrelu': zReLU,
                    'modrelu': modReLU,
                    'cleakyrelu':cLeakyReLU,
                    'ctanh': tanh,
                    'leakyrelu': LeakyReLU,
                    'amu': AmplitudeMaxout,
                    'fleakyrelu': FLeakyReLU,
                    'complexrelu': complexReLU,
                    'mish': mish,
                    'relu': tf.keras.activations.relu
                    }[self.activations]

    def _convert_loss(self):
        # convert "string" to func
        # default_losses = { is complex network:loss list}
        default_losses = {
            True:{'complexrms', 'complexmae', 'complexmse', 'ssim', 'ms_ssim', 'ssim_mse', 'assim_mse', 'ssim_map', 'wmse'},
            False:{'mse','ssim', 'mae', 'ms_ssim', 'ssim_mse', 'assim_mse', 'ssim_map', 'wmse'}
            }[self.complex_network]
        
        if isinstance(self.losses, str):
            self.losses = self.losses.lower()
            if self.losses in default_losses:
                self.losses = {
                    'complexrms': ComplexRMS,
                    'complexmae': ComplexMAE,
                    'complexmse': ComplexMSE,
                    'ssim'      : SSIM,
                    'ms_ssim'   : MS_SSIM,
                    'ssim_mse'  : SSIM_MSE,
                    'assim_mse' : aSSIM_MSE,
                    'ssim_map'  : ssim_map,
                    'wmse'      : wMSE,
                    'mse'       : MSE,
                    'mae'       : MAE                    
                    }[self.losses]
    def _convert_opt(self, opt=None):
        opts = {
            'adam':tf.keras.optimizers.Adam(learning_rate=self.lr, amsgrad=False),
            'adadelta':tf.keras.optimizers.Adadelta(self.lr),
            'sgd':tf.keras.optimizers.SGD(CosineDecay(self.lr, 20, 1e-2), momentum=0.5, nesterov=True),
            'rmsprop':tf.keras.optimizers.RMSprop(self.lr,momentum=0.9)
            }
        if opt is not None:
            if isinstance(opt, str):
                opt = opt.lower()
                return opts[opt]
            else:
                return opt
        if isinstance(self.optimizer, str):
            self.optimizer = self.optimizer.lower()
            self.optimizer = opts[self.optimizer]
            
    def _define_func(self):
        self.convFunc = ComplexConv2D if self.complex_network else Conv2D
        self.bnFunc = ComplexBatchNormalization if self.complex_network else BatchNormalization
        self.concatFunc = ComplexConcatenate if self.complex_network else Concatenate
        self.fine_tune = False
        
    def _cal_estimated_time(self, consump_time_list, cur_iter):
        finish_time = sum(consump_time_list)/cur_iter*(self.epochs - cur_iter) # in sec
        finish_time_hr = int(finish_time//3600)
        finish_time_min = int((finish_time%3600)//60)
        finish_time_sec = int(finish_time - 3600*finish_time_hr - 60*finish_time_min)
        return f"{finish_time_hr} hr {finish_time_min} min {finish_time_sec} sec"
                
    # clc.training -> start training
    def training(self, x, y, model_name=None):
        # prepare for history list of loss curve
        history = {
            'loss':[]
            }
        # convert complex data from complex format to 2-branch real format
        if self.complex_network and np.issubdtype(x.dtype, np.complexfloating):
            # [N,H,W,1] complex64 -> [N,H,W,2] float32
            x_train, y_train = convert_to_real(x), convert_to_real(y)
        else:
            x_train, y_train = x, y
        # check validation data
        # if `validation_data` exist -> use validation data (ignore validation rate)
        hasval = True
        if self.validation_data is not None:
            x_val, y_val = self.validation_data
            # convert complex data from complex format to 2-branch real format
            if self.complex_network and np.issubdtype(x_val.dtype, np.complexfloating):
                x_val, y_val = convert_to_real(x_val), convert_to_real(y_val)
        elif self.validation_rate:
            num_train = round(x.shape[0]*(1-self.validation_rate))
            x_val, y_val = x_train[num_train:], y_train[num_train:]
            x_train, y_train = x_train[:num_train], y_train[:num_train]
        else:
            hasval = False
        # number of training data
        N = x_train.shape[0]
        # shuffle the whole training dataset in each epoch and then it is splited into mini-batch
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(N).batch(self.batch_size)
            
        self.input_shape = x_train.shape[1:] # [H,W,C]
        self._build_info() # construct model information sheet
        # ==============================================================
        if model_name is None:
            self._convert_loss()
            self._convert_act()
            self._convert_opt()
            self._define_func()
            self.model = self.build_model()
        else:
            self.model_info['basedmodel'] = model_name
            # self.model = tf.keras.models.load_model(os.path.join(constant.MODELPATH,model_name,model_name+'.h5'),self.get_custom_object(), False)
            self.model = tf.keras.models.load_model(os.path.join(constant.MODELPATH,model_name,model_name),self.get_custom_object(), False)
            self.fine_tune = True
            self._convert_loss()
            self._convert_act()
        # ==============================================================
        # show model architecture
        self.model.summary()
        # generate model name to be saved
        self._generate_modelname()
        saved_dir = os.path.join(constant.MODELPATH, self.model_name)
        # show B-mode image of one of the psfs in training data and validation data.
        Bmode_fig(y_train[4:5], 80, # 4-th PSF
                      title_name="Ground truth", 
                      show=True,
                      saved_name=os.path.join(saved_dir, 'PredictionEachEpoch',"Ground_truth.png"))
        if hasval:
            valid_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).shuffle(len(x_val)).batch(self.batch_size)
            history['val_loss'] = []
            Bmode_fig(y_val[5:6], 80, # 5-th PSF
                          title_name="Ground truth validation",
                          show=True,
                          saved_name=os.path.join(saved_dir, 'PredictionEachEpoch',"Ground_truth_val.png"))
        consump_times = []
        # training epochs
        for epoch in range(self.epochs):
            if self.batchsizeschedule and epoch != 0:
                batch_size = int(1 + self.batch_size//self.epochs)
                train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(N).batch(batch_size)
            
            s = time.time() # start time for each epoch
            loss_train_epoch = []
            for x_batch_train, y_batch_train in train_dataset:
                loss_train_epoch.append(self.train_step(x_batch_train, y_batch_train))
            history['loss'].append(np.mean(loss_train_epoch))
            if hasval:
                loss_valid_epoch = []
                for x_batch_val, y_batch_val in valid_dataset:
                    loss_valid_epoch.append(self.test_step(x_batch_val, y_batch_val))
                history['val_loss'].append(np.mean(loss_valid_epoch))
                e = time.time() # end time for each epoch
                consump_times.append(e - s)
                print(f'Epoch:{epoch+1}/{self.epochs} - {(e-s):.1f}s - ' \
                      f'loss:{np.mean(loss_train_epoch):.6f} - ' \
                      f'val_loss:{np.mean(loss_valid_epoch):.6f} \n')
            else:
                e = time.time()
                consump_times.append(e - s)
                print(f'Epoch:{epoch+1}/{self.epochs} - {(e-s):.1f}s - ' \
                      f'loss:{np.mean(loss_train_epoch):.6f} - \n')
                    
            if (epoch+1)%5 == 0:
                # check the prediction performance
                train_img = self.model.predict(x_train[:8]) 
                print(f"min value: {np.min(np.abs(train_img))}, avg value: {np.mean(np.abs(train_img))}" \
                      f", max value: {np.max(np.abs(train_img))} \n")
                print(f"Estimated finish time: {self._cal_estimated_time(consump_times, epoch+1)}")
                Bmode_fig(train_img[4:5], 80,
                          title_name=f"Prediction-epoch-{epoch+1}", 
                          show=True,
                          saved_name=os.path.join(saved_dir, 'PredictionEachEpoch',f"epoch-{epoch+1}.png"))
                if hasval:
                    val_img = self.model.predict(x_val[:8])
                    Bmode_fig(val_img[5:6], 80,
                              title_name=f"Validation-epoch-{epoch+1}", 
                              show=True,
                              saved_name=os.path.join(saved_dir, 'PredictionEachEpoch',f"epoch-val-{epoch+1}.png"))
            # save the model weights. maybe you need this one
            if (epoch+1)%500 == 0:
                print('saving...')
                save_path = os.path.join(saved_dir, f'checkpoints/chkpt-epoch{epoch+1:04d}/')
                if os.path.exists(save_path):
                    os.makedirs(save_path)
                self.model.save_weights(save_path)
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
    
    def training_multiple_opt(self, x, y, compared_opt):
        # prepare for history list of loss curve
        history = {
            'loss_optimizer1':[],
            'loss_optimizer2':[]
            }
        # convert complex data from complex format to 2-branch real format
        if self.complex_network and np.issubdtype(x.dtype, np.complexfloating):
            # [N,H,W,1] complex64 -> [N,H,W,2] float32
            x_train, y_train = convert_to_real(x), convert_to_real(y)
        # check validation data
        # if `validation_data` exist -> use validation data (ignore validation rate)
        hasval = True
        if self.validation_data is not None:
            x_val, y_val = self.validation_data
            # convert complex data from complex format to 2-branch real format
            if self.complex_network and np.issubdtype(x_val.dtype, np.complexfloating):
                x_val, y_val = convert_to_real(x_val), convert_to_real(y_val)
        elif self.validation_rate:
            num_train = round(x.shape[0]*(1-self.validation_rate))
            x_val, y_val = x_train[num_train:], y_train[num_train:]
            x_train, y_train = x_train[:num_train], y_train[:num_train]
        else:
            hasval = False
        # number of training data
        N = x_train.shape[0]
        # shuffle the whole training dataset in each epoch and then it is splited into mini-batch
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(N).batch(self.batch_size)
        
        self._convert_loss()
        self._convert_act()
        self.optimizer1 = self._convert_opt(self.optimizer)
        self.optimizer2 = self._convert_opt(compared_opt)
        
        self.model1 = self.build_model()
        self.model2 = self.build_model()
        
        self.input_shape = x_train.shape[1:] # [H,W,C]
        self._build_info() # construct model information sheet
        # show model architecture
        self.model.summary()
        # generate model name to be saved
        self._generate_modelname()
        saved_dir = os.path.join(constant.MODELPATH, self.model_name)
        # show B-mode image of one of the psfs in training data and validation data.
        Bmode_fig(y_train[4:5], 80, # 4-th PSF
                      title_name="Ground truth", 
                      show=True,
                      saved_name=os.path.join(saved_dir, 'PredictionEachEpoch',"Ground_truth.png"))
        if hasval:
            valid_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).shuffle(len(x_val)).batch(self.batch_size)
            history['val_loss_optimizer1'] = []
            history['val_loss_optimizer2'] = []
            Bmode_fig(y_val[5:6], 80, # 5-th PSF
                          title_name="Ground truth validation",
                          show=True,
                          saved_name=os.path.join(saved_dir, 'PredictionEachEpoch',"Ground_truth_val.png"))
        # training epochs
        for epoch in range(self.epochs):
            if self.batchsizeschedule and epoch != 0:
                batch_size = int(1 + self.batch_size//self.epochs)
                train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(N).batch(batch_size)
            
            s = time.time() # start time for each epoch
            loss_train_epoch1 = []
            loss_train_epoch2 = []
            for x_batch_train, y_batch_train in train_dataset:
                losses = self.train_step_multiple(x_batch_train, y_batch_train)
                loss_train_epoch1.append(losses[0])
                loss_train_epoch2.append(losses[1])
            history['loss_optimizer1'].append(np.mean(loss_train_epoch1))
            history['loss_optimizer2'].append(np.mean(loss_train_epoch2))
            if hasval:
                loss_valid_epoch1 = []
                loss_valid_epoch2 = []
                for x_batch_val, y_batch_val in valid_dataset:
                    losses = self.test_step_multiple(x_batch_val, y_batch_val)
                    loss_valid_epoch1.append(losses[0])
                    loss_valid_epoch2.append(losses[1])
                history['val_loss_optimizer1'].append(np.mean(loss_valid_epoch1))
                history['val_loss_optimizer2'].append(np.mean(loss_valid_epoch2))
                e = time.time() # end time for each epoch
                print(f'Epoch:{epoch+1}/{self.epochs} - {(e-s):.1f}s - ' \
                      f'loss1:{np.mean(loss_train_epoch1):.6f} - ' \
                      f'loss2:{np.mean(loss_train_epoch2):.6f} - ' \
                      f'val_loss1:{np.mean(loss_valid_epoch1):.6f} - ' \
                      f'val_loss2:{np.mean(loss_valid_epoch2):.6f} \n')
            else:
                e = time.time()
                print(f'Epoch:{epoch+1}/{self.epochs} - {(e-s):.1f}s - ' \
                      f'loss1:{np.mean(loss_train_epoch1):.6f} - '\
                          f'loss2:{np.mean(loss_train_epoch2):.6f} \n')
            if (epoch+1)%5 == 0:
                train_img1 = self.model1.predict(x_train[:8])
                train_img2 = self.model2.predict(x_train[:8])
                Bmode_fig(normalization(train_img1[4:5]),80,
                          title_name=f"Prediction1-epoch-{epoch+1}", 
                          show=True,
                          saved_name=os.path.join(saved_dir, 'PredictionEachEpoch',f"optimizer1-epoch-{epoch+1}.png"))
                Bmode_fig(normalization(train_img2[4:5]),80,
                          title_name=f"Prediction2-epoch-{epoch+1}", 
                          show=True,
                          saved_name=os.path.join(saved_dir, 'PredictionEachEpoch',f"optimizer2-epoch-{epoch+1}.png"))
                if hasval:
                    val_img1 = self.model1.predict(x_val[:8])
                    val_img2 = self.model2.predict(x_val[:8])
                    Bmode_fig(normalization(val_img1[5:6]),80,
                              title_name=f"Validation1-epoch-{epoch+1}", 
                              show=True,
                              saved_name=os.path.join(saved_dir, 'PredictionEachEpoch',f"optimizer1-epoch-val-{epoch+1}.png"))
                    
                    Bmode_fig(normalization(val_img2[5:6]),80,
                              title_name=f"Validation2-epoch-{epoch+1}", 
                              show=True,
                              saved_name=os.path.join(saved_dir, 'PredictionEachEpoch',f"optimizer2-epoch-val-{epoch+1}.png"))
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
                 filters=16,
                 size=(3,3),
                 batch_size=32,
                 lr=1e-3,
                 epochs=1000,
                 validation_rate=None,
                 validation_data=None,
                 seed=7414,
                 activations='LeakyReLU',
                 losses='SSIM',
                 optimizer='Adam',
                 dropout=True,
                 use_bias=False,
                 complex_network=True,
                 batchsizeschedule=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.size = size
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.validation_rate = validation_rate
        self.validation_data = validation_data
        self.seed = seed
        self.activations = activations
        self.losses = losses
        self.optimizer = optimizer
        self.dropout = dropout
        self.use_bias = use_bias
        self.complex_network = complex_network
        self.batchsizeschedule = batchsizeschedule
        
        
    def build_model(self):
        tf.random.set_seed(self.seed)
        inputs = Input(self.input_shape)
        x = inputs
        # downsample
        skips = []
        filters = [2**ii*self.filters for ii in range(5)] 
        dropout_rates = [None,None,0.3,0.5,0.7] if self.dropout else [None, None, None, None, None]
        for ii in range(5):
            # downsample 5 times, (H,W) -> (H/32,W/32)
            x = self.conv_block(x, filters[ii], self.size, False, self.activations, self.use_bias, strides=2) # downsample block
            x = self.conv_block(x, filters[ii], self.size, True, self.activations, self.use_bias, dropout_rates[ii])
            skips.append(x)
        x = self.conv_block(x, filters[ii], self.size, True, self.activations, self.use_bias, dropout_rates[ii])
        # upsample
        skips = reversed(skips)
        filters = [2**max(0,ii)*self.filters for ii in range(-1,4)] 
        for skip in skips:
            x = self.concatFunc()([x,skip])
            x = self.conv_block(x, filters[ii], self.size, False, self.activations, self.use_bias, strides=2, transposed=True)
            x = self.conv_block(x, filters[ii], self.size, True, self.activations, self.use_bias, dropout_rates[ii])
            ii = ii - 1
        x = self.convFunc(1, (3,3), padding='same', use_bias=self.use_bias)(x)
        x = self.activations()(x)
        return tf.keras.Model(inputs=inputs, outputs=x)
    
class SOSNet(BaseModel):
    def __init__(self,
                 filters=16,
                 size=(3,3),
                 batch_size=32,
                 lr=1e-3,
                 epochs=1000,
                 validation_rate=None,
                 validation_data=None,
                 seed=7414,
                 activations='LeakyReLU',
                 losses='SSIM',
                 optimizer='Adam',
                 dropout=False,
                 use_bias=False,
                 complex_network=True,
                 batchsizeschedule=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.size = size
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.validation_rate = validation_rate
        self.validation_data = validation_data
        self.seed = seed
        self.activations = activations
        self.losses = losses
        self.optimizer = optimizer
        self.dropout = dropout
        self.use_bias = use_bias
        self.complex_network = complex_network
        self.batchsizeschedule = batchsizeschedule
        
        
    def build_model(self):
        tf.random.set_seed(self.seed)
        inputs = Input(self.input_shape)
        x = inputs
        # downsample
        skips = []
        filters = [2**ii*self.filters for ii in range(5)] 
        dropout_rates = [None,None,0.1,0.3,0.5] if self.dropout else [None, None, None, None, None]
        for ii in range(5):
            # downsample 5 times, (H,W) -> (H/32,W/32)
            x = self.conv_block(x, filters[ii], self.size, False, self.activations, self.use_bias, strides=2) # downsample block
            x = self.conv_block(x, filters[ii], self.size, True, self.activations, self.use_bias, dropout_rates[ii])
            skips.append(x)
        x = self.conv_block(x, filters[ii], self.size, True, self.activations, self.use_bias, dropout_rates[ii])
        # upsample
        skips = reversed(skips)
        filters = [2**max(0,ii)*self.filters for ii in range(-1,4)] 
        for skip in skips:
            x = self.concatFunc()([x,skip])
            x = self.conv_block(x, filters[ii], self.size, False, self.activations, self.use_bias, strides=2, transposed=True)
            x = self.conv_block(x, filters[ii], self.size, True, self.activations, self.use_bias, dropout_rates[ii])
            ii = ii - 1
        x = self.convFunc(1, (3,3), padding='same', use_bias=self.use_bias)(x)
        x = self.activations()(x)
        return tf.keras.Model(inputs=inputs, outputs=x)
      
class SegNet(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build_model(self):
        tf.random.set_seed(self.seed)
        inputs = Input(self.input_shape)
        x = inputs
        # downsample
        inds = []
        filters = [2**ii*self.filters for ii in range(5)] 
        dropout_rates = [None,None,0.3,0.5,0.7] if self.dropout else [None, None, None, None, None]
        for ii in range(5):
            # downsample 5 times, (H,W) -> (H/32,W/32)
            x, ind = self.downsample_block_argmax(x, False, self.activations) # downsample block
            x = self.conv_block(x, filters[ii], self.size, True, self.activations, self.use_bias, dropout_rates[ii])
            inds.append(x)
        x = self.conv_block(x, filters[ii], self.size, True, self.activations, self.use_bias, dropout_rates[ii])
        # upsample
        inds = reversed(inds)
        filters = [2**max(0,ii)*self.filters for ii in range(-1,4)]
        for ind in inds:
            x = self.upsample_block_unpool([x,ind], False, self.activations)
            x = self.conv_block(x, filters[ii], self.size, True, self.activations, self.use_bias, dropout_rates[ii])
            ii = ii - 1
        x = self.convFunc(1, (1,1), padding='same', use_bias=self.use_bias)(x)
        x = self.bnFunc()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        return tf.keras.Model(inputs=inputs, outputs=x)
        
class PSNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def build_model(self):
        tf.random.set_seed(self.seed)
        inputs = Input(self.input_shape)          
        x = inputs
        filters = [2**ii*self.filters for ii in range(5)]
        dropout_rates = [None,None,0.3,0.5,0.7] if self.dropout else [None, None, None, None, None]
        # downsample
        for ii in range(5):
            x = self.conv_block(x, filters[ii], self.size, False, self.activations, self.use_bias, strides=2) # downsample block
            x = self.conv_block(x, filters[ii], self.size, True, self.activations, self.use_bias, dropout_rates[ii])
        x = self.conv_block(x, filters[ii], self.size, True, self.activations, self.use_bias, dropout_rates[ii])
        # upsample
        filters = [2**max(0,ii)*self.filters for ii in range(-1,4)]
        for ii in range(5,0,-1):
            x = self.upsample_block_PS(x, False, self.activations)
            x = self.conv_block(x, filters[ii], self.size, True, self.activations, self.use_bias, dropout_rates[ii])
        for _ in range(3):
            x = self.convFunc(2*self.filters, 3, padding='same', use_bias=self.use_bias)(x)
            x = self.bnFunc()(x)
            x = self.activations()(x)
        x = self.convFunc(1, 3, padding='same', use_bias=self.use_bias)(x)
        x = self.activations()(x)
        return tf.keras.Model(inputs=inputs, outputs=x)
    
class ResNet50(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def build_model(self):
        tf.random.set_seed(self.seed)
        inputs = Input(self.input_shape)
        # Stage 1
        x = self.conv_block(inputs, 2*self.filters, (3,3), True, self.activations, self.use_bias)
        x, ind = self.downsample_block_argmax(x)
  

        # Stage 2
        x = self.conv_block(x, self.filters*2, (3,3), True, self.activations, self.use_bias)
        x = self.identity_block(x, self.filters*2, (3,3))
        x = self.identity_block(x, self.filters*2, (3,3), bn=True)
    
    
        # Stage 3 
        x = self.conv_block(x, self.filters*4, (3,3), True, self.activations, self.use_bias)
        x = self.identity_block(x, self.filters*4, (3,3))
        x = self.identity_block(x, self.filters*4, (3,3), bn=True)
        
        # Stage 4 
        x = self.conv_block(x, self.filters*8, (3,3), True, self.activations, self.use_bias)
        x = self.identity_block(x, self.filters*8, (3,3))
        x = self.identity_block(x, self.filters*8, (3,3), bn=True)

        # Stage 5
        x = self.conv_block(x, self.filters*16, (3,3), True, self.activations, self.use_bias)
        x = self.identity_block(x, self.filters*16, (3,3))
        x = self.identity_block(x, self.filters*16, (3,3), bn=True)

        # Stage 6
        x = self.conv_block(x, self.filters*32, (3,3), True, self.activations, self.use_bias)
        x = self.identity_block(x, self.filters*32, (3,3))
        x = self.identity_block(x, self.filters*32, (3,3), bn=True)
        
        # Stage 6
        x = self.conv_block(x, self.filters*64, (3,3), True, self.activations, self.use_bias)
        x = self.identity_block(x, self.filters*64, (3,3))
        x = self.identity_block(x, self.filters*64, (3,3), bn=True)
        
        x = self.conv_block(x, self.filters*2, (3,3), False, self.activations, self.use_bias)
        x = MaxUnpooling2D((2,2))([x, ind])
        x = self.activations()(x)
        # Stage 6
        x = self.conv_block(x, self.filters*2, (3,3), False, self.activations, self.use_bias)
        x = self.convFunc(1, 3, padding='same', use_bias=self.use_bias)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        return tf.keras.Model(inputs=inputs, outputs=x) 

class ResShuffle(BaseModel):
    # reference paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9062600&tag=1
    def __init__(self, *args,**kwargs):
        super().__init__(*args, **kwargs)
        
    def build_model(self):
        tf.random.set_seed(self.seed)
        inputs = Input(self.input_shape)
        # Stage 1
        x = self.conv_block(inputs, self.filters, (3,3), True, self.activations, self.use_bias, strides=2)
        # Stage 2
        x = self.conv_block(x, self.filters, (3,3), True, self.activations, self.use_bias)
        res2 = self.conv_block(x, self.filters, (3,3), True, self.activations, self.use_bias)
        res2 = self.convFunc(4*self.filters, (1, 1), strides = 1, padding='same', use_bias=self.use_bias)(res2)
        # Stage 3
        x = self.conv_block(x, 4*self.filters, (3,3), True, self.activations, self.use_bias)
        x = self.convFunc(4*self.filters, (3, 3), strides = 1, padding='same', use_bias=self.use_bias)(x)
        x = Add()([res2,x])
        x = self.bnFunc()(x)
        res3 = self.activations(x)
        res3 = self.convFunc(8*self.filters, (1, 1), strides = 1, padding='same', use_bias=self.use_bias)(res3)
        # Stage 4
        x = self.conv_block(x, 8*self.filters, (3,3), True, self.activations, self.use_bias)
        x = self.convFunc(8*self.filters, (3, 3), strides = 1, padding='same', use_bias=self.use_bias)(x)
        x = Add()([res3,x])
        x = self.bnFunc()(x)
        res4 = self.activations(x)
        res4 = self.convFunc(16*self.filters, (1, 1), strides = 1, padding='same', use_bias=self.use_bias)(res4)
        # Stage 5
        x = self.conv_block(x, 16*self.filters, (3,3), True, self.activations, self.use_bias)
        x = self.convFunc(16*self.filters, (3, 3), strides = 1, padding='same', use_bias=self.use_bias)(x)
        x = Add()([res4,x])
        x = self.bnFunc()(x)
        res5 = self.activations(x)
        res5 = self.convFunc(32*self.filters, (1, 1), strides = 1, padding='same', use_bias=self.use_bias)(res5)
        # Stage 6
        x = self.convFunc(32*self.filters, (3, 3), strides = 1, padding='same', use_bias=self.use_bias)(x)
        x = Add()([res5, x])
        x = self.bnFunc()(x)
        x = self.activations()(x)
        x = self.convFunc(256, (3, 3), strides = 1, padding='same', use_bias=self.use_bias)(res5)
        x = PixelShuffler((16,16))(x)
        return tf.keras.Model(inputs=inputs, outputs=x)

class ResUNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def build_model(self): 
        tf.random.set_seed(self.seed)
        inputs = Input(self.input_shape)         
        x = inputs
        # downsample
        inds = []
        if self.num_downsample_layer is None:
            self.num_downsample_layer = 5
        x = self.conv_block(x, self.filters, 3, False, self.activations, self.use_bias)
        for ithlayer in range(self.num_downsample_layer):
            filters = self.filters*2**ithlayer
            skip = self.conv_block(x, filters, 1, False, self.activations, self.use_bias, strides=2)
            x, ind = self.downsample_block_argmax(x)
            x = self.conv_block(x, filters, self.size, True, self.activations, self.use_bias)            
            x = Add()([x, skip])
            inds.append(ind)
        x = self.conv_block(x, filters//2, self.size, False, self.activations, self.use_bias)
        # upsample
        inds = reversed(inds)
        for ind in inds:
            filters = self.filters*2**max(ithlayer-2,0)
            skip = self.conv_block(x, filters, 1, False, self.activations, self.use_bias, strides=2, transposed=True)
            x = self.upsample_block_unpool([x, ind])
            x = self.conv_block(x, filters, self.size, True, self.activations, self.use_bias)
            x = Add()([x,skip])
            ithlayer = ithlayer - 1            
        for _ in range(3):
            x = self.conv_block(x, 2*self.filters, 3, True, self.activations, self.use_bias)
        x = self.conv_block(x, 1, 1, True, self.activations, self.use_bias)
  
        x = tf.keras.layers.LeakyReLU()(x)
        return tf.keras.Model(inputs=inputs, outputs=x)
        
class OldUNet(BaseModel):
    # IUS2022 model
    def __init__(self,
                 filters=8,
                 size=(3,3),
                 batch_size=8,
                 lr=1e-3,
                 epochs=200,
                 seed=7414,
                 activations='FLeakyReLU',
                 losses='SSIM',
                 dropout_rates=None,
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
            dropout_rates=dropout_rates,
            use_bias=use_bias,
            complex_network=complex_network,
            num_downsample_layer=num_downsample_layer,
            **kwargs)
        
    def build_model(self):
        tf.random.set_seed(self.seed)
        inputs = Input(self.input_shape)          
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
   
        x = tf.keras.layers.LeakyReLU()(x)
        return tf.keras.Model(inputs=inputs, outputs=x)

class RevisedSegNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def build_model(self):
        tf.random.set_seed(self.seed)
        inputs = Input(self.input_shape)        
        x = inputs
        if self.num_downsample_layer is None:
            self.num_downsample_layer = 5
        # downsample
        inds = []
        if self.dropout is not None:
            dropout_rates = [None]*(self.num_downsample_layer-2) + [0.7, 0.9]
        else:
            dropout_rates = [None]*self.num_downsample_layer
        x = self.conv_block(x, self.filters, 3, False, self.activations, self.use_bias)
        for ithlayer in range(self.num_downsample_layer):
            filters = self.filters*2**ithlayer
            x, ind = self.downsample_block_argmax(x, False, self.activations) # downsampling 2^num_downsample_layer
            if ithlayer > self.num_downsample_layer - 2:
                # use batch normalization
                x = self.conv_block(x, filters, self.size, True, self.activations, self.use_bias, dropout_rates[ithlayer])
            else:
                x = self.conv_block(x, filters, self.size, False, self.activations, self.use_bias, dropout_rates[ithlayer])
            inds.append(ind)
        x = self.conv_block(x, filters//2, self.size, True, self.activations, self.use_bias)
        # upsample
        inds = reversed(inds)
        for ind in inds:
            x = self.upsample_block_unpool([x, ind], False, self.activations)
            filters = self.filters*2**max(ithlayer-2,0)
            if ithlayer > self.num_downsample_layer - 2:
                x = self.conv_block(x, filters, self.size, True, self.activations, self.use_bias, dropout_rates[ithlayer])
            else:
                x = self.conv_block(x, filters, self.size, False, self.activations, self.use_bias, dropout_rates[ithlayer])
            ithlayer = ithlayer - 1
        for _ in range(3):
            x = self.conv_block(x, 2*self.filters, 3, False, self.activations, self.use_bias)
        x = self.convFunc(1, (1,1), padding='same', use_bias=self.use_bias)(x)
 
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
                 use_bias=False,
                 complex_network=True,
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
            use_bias=use_bias,
            complex_network=complex_network,
            **kwargs)
        
    def build_model(self):
        tf.random.set_seed(self.seed)
        inputs = Input(self.input_shape)        
        x = inputs
        x = self.conv_block(x, 4*self.filters, self.size, False, self.activations, self.use_bias) # 64
        # downsample - 1
        x = self.conv_block(x, 4*self.filters, self.size, False, self.activations, self.use_bias, strides=2) # 64
        x = self.conv_block(x, 8*self.filters, self.size, False, self.activations, self.use_bias) # 128
        # downsample - 2
        x = self.conv_block(x, 8*self.filters, self.size, False, self.activations, self.use_bias, strides=2) # 128
        x = self.conv_block(x, 16*self.filters, self.size, True, self.activations, self.use_bias) # 256
        # downsample - 3
        x = self.conv_block(x, 16*self.filters, self.size, False, self.activations, self.use_bias, strides=2) # 256
        skip1 = x
        x = self.conv_block(x, 32*self.filters, self.size, True, self.activations, self.use_bias) # 512
        # downsample - 4
        x = self.conv_block(x, 32*self.filters, self.size, False, self.activations, self.use_bias, strides=2) # 512
        skip2 = x
        x = self.conv_block(x, 64*self.filters, self.size, True, self.activations, self.use_bias) # 1024
        # downsample - 5
        x = self.conv_block(x, 64*self.filters, self.size, False, self.activations, self.use_bias, strides=2) # 1024
        skip3 = x
        x = self.conv_block(x, 64*self.filters, self.size, True, self.activations, self.use_bias) # 1024
        # upsample - 5
        x = self.concatFunc()([x,skip3])
        x = self.conv_block(x, self.filters, self.size, False, self.activations, self.use_bias, strides=2, transposed=True)
        x = self.conv_block(x, self.filters, self.size, True, self.activations, self.use_bias) 
        # upsample - 4
        x = self.concatFunc()([x,skip2])
        x = self.conv_block(x, 2*self.filters, self.size, False, self.activations, self.use_bias, strides=2, transposed=True)
        x = self.conv_block(x, 2*self.filters, self.size, True, self.activations, self.use_bias) 
        # upsample - 3
        x = self.concatFunc()([x,skip1])
        x = self.conv_block(x, 4*self.filters, self.size, False, self.activations, self.use_bias, strides=2, transposed=True)
        x = self.conv_block(x, 4*self.filters, self.size, True, self.activations, self.use_bias) 
        # upsample - 2
        x = self.conv_block(x, 8*self.filters, self.size, False, self.activations, self.use_bias, strides=2, transposed=True)
        x = self.conv_block(x, 8*self.filters, self.size, False, self.activations, self.use_bias) 
        # upsample - 1
        x = self.conv_block(x, 16*self.filters, self.size, False, self.activations, self.use_bias, strides=2, transposed=True)
        x = self.conv_block(x, 16*self.filters, self.size, False, self.activations, self.use_bias) 
        for _ in range(3):
            x = self.conv_block(x, 2*self.filters, self.size, False, self.activations, self.use_bias) 
        x = self.convFunc(1, (1,1), padding='same', use_bias=self.use_bias)(x)
        x = self.activations()(x)
        return tf.keras.Model(inputs=inputs, outputs=x)
        
class SRN(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def res_block(self, x, filters, kernel_size):
        # reference paper: Scale-recurrent Network for Deep Image Deblurring
        # https://arxiv.org/pdf/1802.01770v1.pdf
        shortcut = x
        x = self.conv_block(x, filters, kernel_size, True, self.activations, self.use_bias)
        x = self.convFunc(filters, kernel_size, strides=1, padding='same', use_bias=self.use_bias)(x)
        x = Add()([shortcut,x])
        return x  
        
    def eblock(self, x, filters, kernel_size):
        # encoder ResBlocks
        # reference paper: Scale-recurrent Network for Deep Image Deblurring
        # https://arxiv.org/pdf/1802.01770v1.pdf
        x = self.conv_block(x, filters, kernel_size, False, self.activations, self.use_bias, strides=2)
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
        x = self.conv_block(x, 2*self.filters, self.size, False, self.activations, self.use_bias, strides=2, transposed=True)
        # upsample - decoder ResBlocks
        x = Add()([x, skip2])
        x = self.res_block(x, 2*self.filters, self.size)
        x = self.res_block(x, 2*self.filters, self.size)
        x = self.conv_block(x, self.filters, self.size, False, self.activations, self.use_bias, strides=2, transposed=True)
        # last block - OutBlock
        x = Add()([x, skip1])
        x = self.res_block(x, self.filters, self.size)
        x = self.res_block(x, self.filters, self.size)
        x = self.res_block(x, self.filters, self.size)
        x = self.conv_block(x, 1, self.size, False, self.activations, self.use_bias, strides=2, transposed=True)
        return tf.keras.Model(inputs=inputs, outputs=x)

class TestUNet(BaseModel):
    def __init__(self, 
                 filters=4,
                 size=(3,3),
                 batch_size=32,
                 lr=1e-3,
                 epochs=1000,
                 seed=7414,
                 activations='LeakyReLU',
                 losses='SSIM',
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
            use_bias=use_bias,
            complex_network=complex_network,
            num_downsample_layer=num_downsample_layer,
            **kwargs)
        
    def build_model(self):
        inputs = Input(self.input_shape)
        constraints = tf.keras.constraints.MinMaxNorm(1., 10.)
        x = inputs
        # downsample
        skips = []
        kernel_initializer = 'complex_independent' if self.complex_network else 'glorot_uniform'
        # dropout_rates = [None,None,None,None,None]
        for ii in range(self.num_downsample_layer):
            filters = self.filters*2**ii
            bn = True if ii >2 else False
            dropout_rate = min(0.9, 0.2*ii-0.1) if ii > 2 else None
            x = self.conv_block(x, filters, self.size, False, self.activations, self.use_bias, strides=2) # downsample block
            x = self.conv_block(x, filters, self.size, bn, self.activations, self.use_bias, dropout_rate, kernel_constraints=constraints, kernel_initializer=kernel_initializer)
            skips.append(x)
        x = self.conv_block(x, filters, self.size, True, self.activations, self.use_bias, dropout_rate)

        # upsample
        skips = reversed(skips)
        for skip in skips:
            filters = self.filters*2**max((ii-1),0)
            bn = True if ii >2 else False
            dropout_rate = min(0.9, 0.2*ii-0.1) if ii > 2 else None
            x = self.concatFunc()([x,skip])
            x = self.conv_block(x, filters, self.size, False, self.activations, self.use_bias, kernel_constraints=constraints, kernel_initializer=kernel_initializer, strides=2, transposed=True)
            x = self.conv_block(x, filters, self.size, bn, self.activations, self.use_bias, dropout_rate)
            ii = ii - 1
        for ii in range(1,4):
            x = self.conv_block(x, 2*ii*self.filters, self.size, True, self.activations, self.use_bias, kernel_constraints=constraints, kernel_initializer=kernel_initializer)
        x = self.convFunc(1, (3,3), padding='same', use_bias=self.use_bias, kernel_constraint=constraints, kernel_initializer=kernel_initializer)(x)
        x = self.activations()(x)
        return tf.keras.Model(inputs=inputs, outputs=x)
        
class TestSegNet(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build_model(self):
        inputs = Input(self.input_shape)
        constraints = tf.keras.constraints.MinMaxNorm(1., 10.)
        x = inputs
        # downsample
        inds = []
        kernel_initializer = 'complex_independent' if self.complex_network else 'glorot_uniform'
        if self.num_downsample_layer is None:
            self.num_downsample_layer = 5
        x = self.conv_block(x, self.filters, 3, False, self.activations, self.use_bias)
        for ii in range(self.num_downsample_layer):
            filters = self.filters*2**ii
            bn = True if ii >2 else False
            dropout_rate = min(0.9, 0.2*ii-0.1) if ii > 2 else None
            x, ind = self.downsample_block_argmax(x, False, self.activations) # downsample block
            x = self.conv_block(x, filters, self.size, bn, self.activations, self.use_bias, dropout_rate, kernel_constraints=constraints, kernel_initializer=kernel_initializer)
            inds.append(ind)
        x = self.conv_block(x, filters//2, self.size, True, self.activations, self.use_bias, dropout_rate)
        # upsample
        inds = reversed(inds)
        for ind in inds:
            filters = self.filters*2**max(ii-2,0)
            bn = True if ii >2 else False
            dropout_rate = min(0.9, 0.2*ii-0.1) if ii > 2 else None
            x = self.upsample_block_unpool([x, ind], False, self.activations)
            x = self.conv_block(x, filters, self.size, bn, self.activations, self.use_bias, dropout_rate)
            ii = ii - 1
        for ii in range(1,4):
            x = self.conv_block(x, 2*ii*self.filters, self.size, True, self.activations, self.use_bias, kernel_constraints=constraints, kernel_initializer=kernel_initializer)
        x = self.convFunc(1, (3,3), padding='same', use_bias=self.use_bias, kernel_constraint=constraints, kernel_initializer=kernel_initializer)(x)
  
        x = self.activations()(x)
        return tf.keras.Model(inputs=inputs, outputs=x)
        
class TestPSNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        
    def build_model(self):
        inputs = Input(self.input_shape)
        constraints = tf.keras.constraints.MinMaxNorm(1., 10.)

        x = inputs
        # downsample
        kernel_initializer = 'complex_independent' if self.complex_network else 'glorot_uniform'
        if self.num_downsample_layer is None:
            self.num_downsample_layer = 5
        # dropout_rates = [None,None,None,None,None]
        for ii in range(self.num_downsample_layer):
            filters = self.filters*2**ii
            bn = True if ii >2 else False
            dropout_rate = min(0.9, 0.2*ii-0.1) if ii > 2 else None
            x = self.conv_block(x, filters, self.size, False, self.activations, self.use_bias, strides=2)
            x = self.conv_block(x, filters, self.size, bn, self.activations, self.use_bias, dropout_rate, kernel_constraints=constraints, kernel_initializer=kernel_initializer)

        x = self.conv_block(x, filters, self.size, True, self.activations, self.use_bias, dropout_rate)
        # upsample
        for ii in range(self.num_downsample_layer,0,-1):
            filters = self.filters*2**max((ii-1),0)
            bn = True if ii > 2 else False
            dropout_rate = min(0.9, 0.2*ii-0.1) if ii > 2 else None
            x = self.upsample_block_PS(x, False, self.activations)
            x = self.conv_block(x, filters, self.size, bn, self.activations, self.use_bias, dropout_rate)
            ii = ii - 1
        for ii in range(1,4):
            x = self.conv_block(x, 2*ii*self.filters, self.size, True, self.activations, self.use_bias, kernel_constraints=constraints, kernel_initializer=kernel_initializer)
        x = self.convFunc(1, (3,3), padding='same', use_bias=self.use_bias, kernel_constraint=constraints, kernel_initializer=kernel_initializer)(x)
        x = self.activations()(x)
        return tf.keras.Model(inputs=inputs, outputs=x)
        
class AMUNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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
        tf.random.set_seed(self.seed)
        inputs = Input(self.input_shape)
        x = inputs
        if self.num_downsample_layer is None:
            self.num_downsample_layer = 5
        # downsample
        skips = []
        for ii in range(self.num_downsample_layer):
            # downsample 5 times, (H,W) -> (H/32,W/32)
            filters = self.filters*2**ii
            bn = True if ii >2 else False
            dropout_rate = min(0.9, 0.2*ii-0.1) if ii > 2 else None
            x = self.conv_block(x, filters, self.size, False, self.activations, self.use_bias, strides=2) # downsample block
            x = self.conv_block(x, filters, self.size, True, self.activations, self.use_bias, dropout_rate)
            skips.append(x)
        x = self.conv_block(x, filters, self.size, True, self.activations, self.use_bias, dropout_rate)
        # upsample
        skips = reversed(skips)
        for skip in skips:
            filters = self.filters*2**max((ii-1),0)
            bn = True if ii > 2 else False
            dropout_rate = min(0.9, 0.2*ii-0.1) if ii > 2 else None
            x = self.concatFunc()([x,skip])
            if ii > 2:
                x = self.conv_block(x, filters, self.size, False, self.activations, self.use_bias, strides=2, transposed=True)
            else:
                x = self.upsample_block_PS(x, False, self.activations)
            x = self.conv_block(x, filters, self.size, bn, self.activations, self.use_bias, dropout_rate)
            ii = ii - 1
        x = self.convFunc(1, (3,3), padding='same', use_bias=self.use_bias)(x)

        x = self.activations()(x)
        return tf.keras.Model(inputs=inputs, outputs=x)
    
class UNet3plus(BaseModel):
    # ref: https://github.com/hamidriasat/UNet-3-Plus/blob/unet3p_lits/models/unet3plus.py
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
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
    
    def conv_block(self, inputs, filters, kernel, strides):
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
    
    def bottleneck(self, inputs, filters, kernel, t, s, r=False):
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
    
        x = self.conv_block(inputs, tchannel, (1, 1), (1, 1))
    
        x = tf.keras.layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
        x = self.bnFunc()(x)
        x = self.relu6()(x)
    
        x = self.convFunc(filters, (1, 1), strides=(1, 1), padding='same')(x)
        x = self.bnFunc()(x)
    
        if r:
            x = tf.keras.layers.add([x, inputs])
        return x
    
    def inverted_residual_block(self, inputs, filters, kernel, t, strides, n):
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
    
        x = self.bottleneck(inputs, filters, kernel, t, strides)
    
        for i in range(1, n):
            x = self.bottleneck(x, filters, kernel, t, 1, True)
    
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
        x = self.conv_block(inputs, 32, (3, 3), strides=(2, 2))
    
        x = self.inverted_residual_block(x, 16, (3, 3), t=1, strides=1, n=1)
        x = self.inverted_residual_block(x, 24, (3, 3), t=6, strides=2, n=2)
        x = self.inverted_residual_block(x, 32, (3, 3), t=6, strides=2, n=3)
        x = self.inverted_residual_block(x, 64, (3, 3), t=6, strides=2, n=4)
        x = self.inverted_residual_block(x, 96, (3, 3), t=6, strides=1, n=3)
        x = self.inverted_residual_block(x, 160, (3, 3), t=6, strides=2, n=3)
        x = self.inverted_residual_block(x, 320, (3, 3), t=6, strides=1, n=1)
    
        x = self.conv_block(x, 1280, (1, 1), strides=(1, 1))
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Reshape((1, 1, 1280))(x)
        x = tf.keras.layers.Dropout(0.8, name='Dropout')(x)
        x = tf.keras.layers.Conv2D(128, (1, 1), padding='same')(x)
        x = tf.keras.layers.Activation('softmax', name='final_activation')(x)
        output = tf.keras.layers.Dense(128)(x)
        return tf.keras.models.Model(inputs=inputs, outputs=output)

class CVCNN(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def conv1d_block(self, x, filters, kernel, bn=True, act=None, dropout_rate=None, strides=1, padding='same', transposed=False):
        if transposed:
            x = tf.keras.layers.Conv1DTranspose(filters, kernel, strides, padding)(x)
        else:
            x = tf.keras.layers.Conv1D(filters, kernel, strides, padding)(x)
        return self.bn_act_drop_block(x, bn, act, dropout_rate)
        
    def _conv_block1(self, x):
        filters = [256,128,32]
        kernel_size = [(3,3),(2,3),(1,1)]
        for ii in range(3):
            x = self.conv_block(x, filters[ii], kernel_size[ii], True, tf.keras.layers.ReLU, strides=2)
        x = Dropout(0.5)(x)
        return x
    
    def _conv_block2(self, x):
        x = self.conv_block(x, 64, (3,3), True, tf.keras.layers.ReLU, strides=2)
        x = self.conv_block(x, 32, (4,4), True, tf.keras.layers.ReLU, strides=2)
        x = self.conv_block(x, 32, (1,1), True, tf.keras.layers.ReLU, strides=2)
        x = Dropout(0.5)(x)
        x = self.conv_block(x, 32, (1,1), act=tf.keras.layers.GlobalAveragePooling2D, strides=2)
        return x
    
    def _inception(self, inputs):
        # block 1
        x1 = self.conv_block(inputs, 48, 1, True, tf.keras.layers.ReLU)
        x1 = self.conv_block(x1, 64, 5, True, tf.keras.layers.ReLU)
        # block 2
        x2 = self.conv_block(inputs, 64, 1, True, tf.keras.layers.ReLU)
        x2 = self.conv_block(x2, 96, 3, True, tf.keras.layers.ReLU)
        x2 = self.conv_block(x2, 96, 3, True, tf.keras.layers.ReLU)
        # block 3
        x3 = self.conv_block(inputs, 64, 1, True, tf.keras.layers.ReLU)
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
        # downsample
        x = self.conv1d_block(x, 16, 3, True, tf.keras.layers.ReLU, None, 2)
        x = self.conv1d_block(x, 32, 3, True, tf.keras.layers.ReLU, None, 2)
        skip1 = x
        x = self.conv1d_block(x, 32, 3, True, tf.keras.layers.ReLU, None, 2)
        skip2 = x
        x = self.conv1d_block(x, 64, 3, True, tf.keras.layers.ReLU, None, 2)
        x = self.conv1d_block(x, 64, 3, True, tf.keras.layers.ReLU)
        # upsample
        x = self.conv1d_block(x, 32, 3, True, tf.keras.layers.ReLU, None, 2, transposed=True)
        x = self.conv1d_block(x, 32, 3, True, tf.keras.layers.ReLU)
        x = self.concatFunc()([x,skip2])
        
        x = self.conv1d_block(x, 16, 3, True, tf.keras.layers.ReLU, None, 2, transposed=True)
        x = self.conv1d_block(x, 16, 3, True, tf.keras.layers.ReLU)
        x = self.concatFunc()([x,skip1])
        # Deconv
        x = self.conv1d_block(x, 8, 3, True, tf.keras.layers.ReLU)
        x = self.conv1d_block(x, 8, 3, True, tf.keras.layers.ReLU, None, 2, transposed=True)
        x = self.conv1d_block(x, 2, 3, True, tf.keras.layers.ReLU)
        x = self.conv1d_block(x, 2, 3, True, tf.keras.layers.ReLU, None, 2, transposed=True)
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
    if not isinstance(model_type, str):
        raise TypeError("`model_type` should be in string.")
    model_type = model_type.upper() # convert to capital
    if model_type in {'UNET'}:
        return UNet
    if model_type in {'SOSNET'}:
        return SOSNet
    elif model_type in {'SEGNET'}:
        return SegNet
    elif model_type in {'PSNET'}:
        return PSNet
    elif model_type in {'RESNET', 'RESNET50'}:
        return ResNet50
    elif model_type in {'RESSHUFFLE'}:
        return ResShuffle
    elif model_type in {'RESUNET'}:
        return ResUNet
    elif model_type in {'OLDNET'}:
        return OldUNet
    elif model_type in {'REVISEDSEGNET'}:
        return RevisedSegNet
    elif model_type in {'REVISEDUNET'}:
        return RevisedUNet
    elif model_type in {'SRN'}:
        return SRN
    elif model_type in {'TEST','TESTUNET'}:
        return TestUNet
    elif model_type in {'TESTSEGNET'}:
        return TestSegNet
    elif model_type in {'TESTPSNET'}:
        return TestPSNet
    elif model_type in {'AMUNET'}:
        return AMUNet
    elif model_type in {'UNET3PLUS'}:
        return UNet3plus
    elif model_type in {'MOBILENET'}:
        return  mobilenetv2
    elif model_type in {'CVCNN'}:
        return CVCNN