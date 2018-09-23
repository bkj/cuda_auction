#!/usr/bin/env python

"""
    make-data.py
"""

import sys
import argparse
import numpy as np
from time import time
from lap import lapjv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=1024)
    parser.add_argument('--max-value', type=int, default=100)
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    
    X = np.random.choice(args.max_value, (args.dim, args.dim))
    
    t = time()
    _, src_ass, _ =  lapjv((X.max() - X))
    jv_time  = int(1000 * (time() - t))
    jv_score = X[(np.arange(X.shape[0]), src_ass)].sum()
    
    print({
        "jv_time"  : jv_time,
        "jv_score" : jv_score,
    }, file=sys.stderr)
    
    print('\n'.join(np.hstack(X).astype('str')))
