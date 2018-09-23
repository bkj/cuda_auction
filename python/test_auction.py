#!/usr/bin/env python

"""
    test-auction.py
"""

from __future__ import print_function

import sys
assert sys.version_info.major == 3, "sys.version_info.major != 3"

import argparse
import numpy as np
from time import time
from lap import lapjv

from lap_auction import dense_lap_auction, sparse_lap_auction

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=1000)
    parser.add_argument('--k', type=int, default=1000)
    parser.add_argument('--max-value', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    
    _ = dense_lap_auction(np.random.choice(10, (10, 10)))
    
    X = np.random.choice(args.max_value + 1, (args.dim, args.dim))
    
    # --
    # Run JV algorithm
    
    print('-' * 50, file=sys.stderr)
    t = time()
    _, src_ass, _ =  lapjv((X.max() - X))
    jv_time  = int(1000 * (time() - t))
    assigned = X[(np.arange(X.shape[0]), src_ass)]
    jv_worst = (X >= assigned.reshape(-1, 1)).sum(axis=1).max()
    jv_score = assigned.sum()
    
    print({
        "jv_time"  : jv_time,
        "jv_worst" : jv_worst,
        "jv_score" : jv_score,
    }, file=sys.stderr)
    
    # --
    # Run dense GPU auction
    
    print('-' * 50, file=sys.stderr)
    t = time()
    auc_ass = dense_lap_auction(X, 
        verbose=True, num_runs=3,
        auction_max_eps=1.0, auction_min_eps=1.0, auction_factor=0.0)
    dense_auction_time  = int(1000 * (time() - t))
    dense_auction_score = int(X[(np.arange(X.shape[0]), auc_ass)].sum())
    
    print({
        "dense_auction_time"  : dense_auction_time,
        "dense_auction_score" : dense_auction_score,
    }, file=sys.stderr)

    # --
    # Run dense GPU auction
    
    print('-' * 50, file=sys.stderr)
    t = time()
    auc_ass = sparse_lap_auction(X, k=args.k, 
        verbose=True, num_runs=3,
        auction_max_eps=1.0, auction_min_eps=1.0, auction_factor=0.0)
    sparse_auction_time  = int(1000 * (time() - t))
    assert len(set(auc_ass)) == args.dim
    sparse_auction_score = int(X[(np.arange(X.shape[0]), auc_ass)].sum())
    
    print({
        "sparse_auction_time"  : sparse_auction_time,
        "sparse_auction_score" : sparse_auction_score,
    }, file=sys.stderr)

