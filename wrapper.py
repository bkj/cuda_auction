import sys
import numpy as np
from time import time
from lapjv import lapjv

import ctypes
from ctypes import c_float, c_int, POINTER

def __get_fn():
    dll = ctypes.CDLL('./cuda_auction.so', mode=ctypes.RTLD_GLOBAL)
    fn = dll.run_auction
    fn.argtypes = [POINTER(c_float), POINTER(c_int), c_int, 
        c_int, c_int, c_float, c_float, c_float]
    
    return fn

__fn = __get_fn()

def run_test(X):
    num_nodes    = int(X.shape[0])
    person2item  = np.zeros(num_nodes).astype('int32')
    
    X_flat = np.hstack(X).astype('float32')
    __fn(
        X_flat.ctypes.data_as(POINTER(c_float)),
        person2item.ctypes.data_as(POINTER(c_int)),
        num_nodes,
        int(1),   # verbose
        int(2),   # num_runs
        float(1.0), # auction_max_eps
        float(1.0 / 4.0), # auction_min_eps
        float(1.0 / 2.0), # auction_factor (0 means end immediately)
    )
    return person2item

# Warm GPU
np.random.seed(123)
_ = run_test(np.random.choice(10, (10, 10)).astype('float32'))

max_value = 100
N         = 2 ** 12
X = np.random.choice(max_value, (N, N)).astype('float32')

t = time()
jv_ass, _, _ = lapjv(X.max() - X)
print('jv score', X[(np.arange(X.shape[0]), jv_ass)].sum(), file=sys.stderr)
print('jv time', time() - t)

t = time()
auc_ass = run_test(X)
print('auction score', X[(auc_ass, np.arange(X.shape[0]))].sum(), file=sys.stderr)
print('auction time', time() - t)



