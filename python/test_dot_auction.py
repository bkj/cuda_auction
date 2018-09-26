#!/usr/bin/env python

"""
    test_dot_auction.py
"""

from __future__ import print_function

import sys
assert sys.version_info.major == 3, "sys.version_info.major != 3"

import sys
import json
import argparse
import numpy as np
from time import time
from scipy import sparse
from lap import lapjv

from lap_auction import dot_auction
from lap_auction import sparse_lap_auction, dense_lap_auction


# --
# Run


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--density', type=float, default=0.5)
    parser.add_argument('--max-value', type=float, default=10)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    if args.k is None:
        args.k = args.dim
    return args

if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    
    _ = dense_lap_auction(np.random.choice(10, (10, 10)))
    
    # --
    # Generate data
    
    t = time()
    A = sparse.rand(args.dim, args.dim, density=args.density, format='csr')
    B = sparse.rand(args.dim, args.dim, density=args.density, format='csr')
    A.eliminate_zeros()
    B.eliminate_zeros()
    assert A.sum(axis=-1).min() > 0
    assert B.sum(axis=-1).min() > 0
    
    gen_time = time() - t
    print('gen_time  ', gen_time, file=sys.stderr)
    
    # X = np.asarray(A.dot(B).todense())
    # print(np.sort(X, axis=-1)[:,::-1])
    
    # --
    # Run JV algorithm
    
    print('-' * 50, file=sys.stderr)
    t = time()
    X = np.asarray(A.dot(B).todense())
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
    
    assert args.k >= jv_worst, "args.k too small"
    
    # --
    # Run dense GPU auction
    
    print('-' * 50, file=sys.stderr)
    t = time()
    X = np.asarray(A.dot(B).todense())
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
    # Run sparse GPU auction
    
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
    
    # --
    # Run auction algorithm
    
    t = time()
    auc_ass = dot_auction(A, B, args.k)
    dot_auction_time  = int(1000 * (time() - t))
    dot_auction_score = int(X[(np.arange(X.shape[0]), auc_ass)].sum())
    
    print({
        "dot_auction_time"  : dot_auction_time,
        "dot_auction_score" : dot_auction_score,
    }, file=sys.stderr)