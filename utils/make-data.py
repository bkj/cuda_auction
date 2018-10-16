#!/usr/bin/env python

"""
    make-data.py
"""

import sys
import argparse
import numpy as np
import pandas as pd
from time import time
from lap import lapjv
from scipy import sparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=1024)
    parser.add_argument('--max-value', type=int, default=100)
    parser.add_argument('--density', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--sparse', action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    
    X = sparse.random(args.dim, args.dim, args.density, dtype=np.float32)
    X = X.tocsr()
    X.data *= args.max_value
    
    t = time()
    Xd = np.asarray(X.todense())
    _, src_ass, _ =  lapjv((Xd.max() - Xd))
    jv_time  = int(1000 * (time() - t))
    jv_score = X[(np.arange(X.shape[0]), src_ass)].sum()
    
    print({
        "jv_time"  : jv_time,
        "jv_score" : jv_score,
    }, file=sys.stderr)
    
    # print('\n'.join(np.hstack(X).astype('str')))
    X_coo = X.tocoo()
    df = pd.DataFrame({"row" : X_coo.row, "col" : X_coo.col, "data" : X_coo.data})
    df.to_csv('./graph', header=None, sep=' ', index=False)
