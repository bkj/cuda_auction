#!/usr/bin/env python

"""
    auction.py
"""

import numpy as np

import ctypes
from ctypes import c_float, c_int, POINTER

def __get_fn():
    dll = ctypes.CDLL('../lib/cuda_auction.so', mode=ctypes.RTLD_GLOBAL)
    fn = dll.run_auction
    fn.argtypes = [POINTER(c_float), POINTER(c_int), c_int, 
        c_int, c_int, c_float, c_float, c_float]
    
    return fn

__fn = __get_fn()

def lap_auction(X, verbose=False, num_runs=1, auction_max_eps=1.0, auction_min_eps=1.0, auction_factor=0.0):
    
    X_flat       = np.hstack(X).astype('float32')
    num_nodes    = int(X.shape[0])
    person2item  = np.zeros(num_nodes).astype('int32')
    
    X_flat_p      = X_flat.ctypes.data_as(POINTER(c_float))
    person2item_p = person2item.ctypes.data_as(POINTER(c_int))
    __fn(
        X_flat_p,
        person2item_p,
        num_nodes,
        int(verbose),
        int(num_runs),
        float(auction_max_eps),
        float(auction_min_eps),
        float(auction_factor),
    )
    return person2item