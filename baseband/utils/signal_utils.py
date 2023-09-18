# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 21:49:20 2023

@author: benzener
"""
# reference: https://github.com/mubeta06/python/blob/master/signal_processing/sp/firwin.py
import numpy


def hamming(M):
    """Return an M + 1 point symmetric hamming window."""
    if M%2:
        raise Exception('M must be even')
    return 0.54 - 0.46*numpy.cos(2*numpy.pi*numpy.arange(M + 1)/M)

def sinc_filter(M, fc):
    """Return an M + 1 point symmetric point sinc kernel with normalised cut-off 
    frequency fc 0->0.5."""
    if M%2:
        raise Exception('M must be even')
    return numpy.sinc(2*fc*(numpy.arange(M + 1) - M/2))

def firwin(numtaps, cutoff):
    h = sinc_filter(numtaps, cutoff)*hamming(numtaps)
    return h/h.sum()