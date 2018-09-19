#!/usr/bin/env python

"""
    test_auction.py
"""

import sys
import numpy as np
from time import time
from lapjv import lapjv

from auction import lap_auction


if __name__ == '__main__':
    # Warm GPU
    np.random.seed(123)
    _ = lap_auction(np.random.choice(10, (10, 10)).astype('float32'))

    max_value = 500
    N         = 2 ** 14
    X = np.random.choice(max_value, (N, N)).astype('float32')
    
    # Run reference implementation
    t = time()
    jv_ass, _, _ = lapjv(X.max() - X)
    
    jv_time  = int(1000 * (time() - t))
    jv_score = int(X[(np.arange(X.shape[0]), jv_ass)].sum())
    print('jv score', jv_score, file=sys.stderr)
    print('jv time', jv_time)
    
    # Run GPU auction
    t = time()
    auc_ass  = lap_auction(X, verbose=True, auction_max_eps=1.0, auction_min_eps=1.0, auction_factor=0.0)
    
    auction_time  = int(1000 * (time() - t))
    auction_score = int(X[(auc_ass, np.arange(X.shape[0]))].sum())
    print('auction score', auction_score, file=sys.stderr)
    print('auction time', auction_time)