#!/usr/bin/env python

"""
    lap_auction.py
"""

import numpy as np
from time import time

import ctypes
from ctypes import c_float, c_double, c_int, POINTER

def __get_run_auction(so_path='/home/bjohnson/projects/cuda_auction/lib/cuda_auction.so'):
    dll = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
    fn = dll.run_auction_python
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

def __get_dot_auction(so_path='/home/bjohnson/projects/cuda_auction/lib/cuda_auction.so'):
    dll = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
    fn = dll.dot_auction
    fn.argtypes = [
        c_int,
        
        POINTER(c_int),
        POINTER(c_int),
        POINTER(c_double),
        
        POINTER(c_int),
        POINTER(c_int),
        POINTER(c_double),
        
        c_int,
        
        POINTER(c_int),
        
        c_int,
    ]
    
    return fn

__run_auction = __get_run_auction()
__dot_auction = __get_dot_auction()

# =============================================================

def dot_auction(A, B, k, verbose=True):
    assert A.shape[0] == B.shape[0]
    assert A.shape[1] == B.shape[1]
    
    num_nodes = int(A.shape[0])
    
    A_indptr  = A.indptr.astype('int32')
    A_indices = A.indices.astype('int32')
    A_data    = A.data.astype('float64')
    
    B_indptr  = B.indptr.astype('int32')
    B_indices = B.indices.astype('int32')
    B_data    = B.data.astype('float64')
    
    A_indptr_p  = A_indptr.ctypes.data_as(POINTER(c_int))
    A_indices_p = A_indices.ctypes.data_as(POINTER(c_int))
    A_data_p    = A_data.ctypes.data_as(POINTER(c_double))
    
    B_indptr_p  = B.indptr.ctypes.data_as(POINTER(c_int))
    B_indices_p = B.indices.ctypes.data_as(POINTER(c_int))
    B_data_p    = B.data.ctypes.data_as(POINTER(c_double))
    
    person2item   = np.zeros(num_nodes).astype('int32')
    person2item_p = person2item.ctypes.data_as(POINTER(c_int))
    
    assert person2item.dtype == np.int32
    __dot_auction(
        num_nodes,
        
        A_indptr_p,
        A_indices_p,
        A_data_p,
        
        B_indptr_p,
        B_indices_p,
        B_data_p,
        
        int(k),
        
        person2item_p,
        
        int(verbose)
    )
    return person2item

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
    
    __run_auction(
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
    
    columns = np.argpartition(X, -k)[:,-k:][:,::-1]
    # columns = np.argsort(-X, axis=-1)[:,:k] # could just `argpartition`
    columns = np.hstack(columns).astype('int32')
    
    assert len(set(columns)) == num_nodes
    # !! If this fails, we can maybe pivot unrepresented columns
    # into the solution.  If that fails, either increase k or 
    # compute some of the entries on the fly. Or just leave some
    # nodes unassigned, and assign them after.
    #
    # If using `topdot`, probably good to have equal elements 
    # kept/dropped randomly
    
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
    
    __run_auction(
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


def csr_lap_auction(X, verbose=False, num_runs=1, 
    auction_max_eps=1.0, auction_min_eps=1.0, auction_factor=0.0):
    
    t = time()
    num_nodes = int(X.shape[0])
    num_edges = X.nnz
    
    # >>
    X.sort_indices()
    # <<
    
    data    = X.data.astype('float32')
    offsets = X.indptr.astype('int32')
    columns = X.indices.astype('int32')
    
    person2item  = np.zeros(num_nodes).astype('int32')
    
    assert data.dtype == np.float32
    assert offsets.dtype == np.int32
    assert columns.dtype == np.int32
    assert person2item.dtype == np.int32
    
    data_p        = data.ctypes.data_as(POINTER(c_float))
    offsets_p     = offsets.ctypes.data_as(POINTER(c_int))
    columns_p     = columns.ctypes.data_as(POINTER(c_int))
    person2item_p = person2item.ctypes.data_as(POINTER(c_int))
    __run_auction(
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
        int(10)
    )
    return person2item