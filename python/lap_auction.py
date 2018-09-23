#!/usr/bin/env python

"""
    lap_auction.py
"""

import numpy as np

import ctypes
from ctypes import c_float, c_int, POINTER

def __get_fn():
    dll = ctypes.CDLL('../lib/cuda_auction.so', mode=ctypes.RTLD_GLOBAL)
    fn = dll.run_auction
    fn.argtypes = [
        c_int,
        c_int,
        
        POINTER(c_float),
        POINTER(c_int),
        POINTER(c_int),
        
        POINTER(c_int),
        
        c_float,
        c_float,
        c_float,
        
        c_int,
        c_int,
    ]
    
    return fn

__fn = __get_fn()

def dense_lap_auction(X, verbose=False, num_runs=1, 
    auction_max_eps=1.0, auction_min_eps=1.0, auction_factor=0.0):
    
    num_nodes    = int(X.shape[0])
    num_edges    = num_nodes * num_nodes
    
    X_flat  = np.hstack(X).astype('float32')
    offsets = np.hstack([num_nodes * np.arange(num_nodes + 1)]).astype('int32')
    columns = (np.arange(num_edges) % num_nodes).astype('int32')
    
    person2item  = np.zeros(num_nodes).astype('int32')
    
    assert X_flat.dtype == np.float32
    assert offsets.dtype == np.int32
    assert columns.dtype == np.int32
    assert person2item.dtype == np.int32
    
    data_p        = X_flat.ctypes.data_as(POINTER(c_float))
    offsets_p     = offsets.ctypes.data_as(POINTER(c_int))
    columns_p     = columns.ctypes.data_as(POINTER(c_int))
    person2item_p = person2item.ctypes.data_as(POINTER(c_int))
    
    __fn(
        num_nodes,
        num_edges,
        
        data_p,
        offsets_p,
        columns_p,
        
        person2item_p,
        
        float(auction_max_eps),
        float(auction_min_eps),
        float(auction_factor),
        
        int(num_runs),
        int(verbose),
    )
    return person2item


def sparse_lap_auction(X, k, verbose=False, num_runs=1, 
    auction_max_eps=1.0, auction_min_eps=1.0, auction_factor=0.0):
    
    num_nodes = int(X.shape[0])
    num_edges = num_nodes * k
    
    offsets = np.hstack([k * np.arange(num_nodes + 1)]).astype('int32')
    
    # columns = np.argpartition(X, -k, axis=-1)[:,-k:]
    columns = np.argsort(X, axis=-1)[:,::-1]
    columns = columns[:,:k]
    columns = np.hstack(columns).astype('int32')
    
    assert len(set(columns)) == num_nodes
    
    sel = np.repeat(np.arange(X.shape[0]), k)
    X_flat = np.ascontiguousarray(X[(sel, columns)].astype('float32'))
    person2item  = np.zeros(num_nodes).astype('int32')
    
    assert X_flat.dtype == np.float32
    assert offsets.dtype == np.int32
    assert columns.dtype == np.int32
    assert person2item.dtype == np.int32
    
    data_p        = X_flat.ctypes.data_as(POINTER(c_float))
    offsets_p     = offsets.ctypes.data_as(POINTER(c_int))
    columns_p     = columns.ctypes.data_as(POINTER(c_int))
    person2item_p = person2item.ctypes.data_as(POINTER(c_int))
    
    __fn(
        num_nodes,
        num_edges,
        
        data_p,
        offsets_p,
        columns_p,
        
        person2item_p,
        
        float(auction_max_eps),
        float(auction_min_eps),
        float(auction_factor),
        
        int(num_runs),
        int(verbose),
    )
    return person2item




