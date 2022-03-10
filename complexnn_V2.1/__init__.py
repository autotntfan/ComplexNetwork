#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Authors: Olexa Bilaniuk
#
# What this module includes by default:
# from . import bn, conv, dense, init, norm, pool
# from . import fft

from . import bn_test, conv_test, dense_test, init_test, norm_test, pool_test
from . import fft_test
from . import activation, loss
# from .bn import ComplexBatchNormalization as ComplexBN
# from .conv import (
#     ComplexConv,
#     ComplexConv1D,
#     ComplexConv2D,
#     ComplexConv3D,
#     WeightNorm_Conv,
# )
# from .dense import ComplexDense
# from .fft import (fft, ifft, fft2, ifft2, FFT, IFFT, FFT2, IFFT2)
# from .init import (
#     ComplexIndependentFilters,
#     IndependentFilters,
#     ComplexInit,
#     SqrtInit,
# )
# from .norm import LayerNormalization, ComplexLayerNorm
# from .pool import SpectralPooling1D, SpectralPooling2D
# from .utils import (
#     get_realpart,
#     get_imagpart,
#     getpart_output_shape,
#     GetImag,
#     GetReal,
#     GetAbs,
# )
from .bn_test import ComplexBatchNormalization as ComplexBN
from .conv_test import (
    ComplexConv,
    ComplexConv1D,
    ComplexConv2D,
    ComplexConv3D,
    WeightNorm_Conv,
)
from .dense_test import ComplexDense
from .fft_test import (fft, ifft, fft2, ifft2, FFT, IFFT, FFT2, IFFT2)
from .init_test import (
    ComplexIndependentFilters,
    IndependentFilters,
    ComplexInit,
    SqrtInit,
)

from .norm_test import LayerNormalization, ComplexLayerNorm
from .pool_test import SpectralPooling1D, SpectralPooling2D
from .utils_test import (
    get_realpart,
    get_imagpart,
    getpart_output_shape,
    GetImag,
    GetReal,
    GetAbs,
)

from .activation import AmplitudeMaxout, cReLU, zReLU, modReLU
from .loss import ComplexRMS, ComplexMSE, ComplexMAE