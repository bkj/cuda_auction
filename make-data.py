#!/usr/bin/env python

"""
    make-data.py
"""

import numpy as np
from time import time
from lapjv import lapjv as jv_src

np.random.seed(123)

N = 2 ** 13
print(N)
max_value = 10000
x = np.random.choice(max_value, (N, N))

t = time()
src_ass, trg_ass, _ =  jv_src((x.max() - x))
trg_ass[:10]
milliseconds = int(1000 * (time() - t))
print(milliseconds)

x[(np.arange(x.shape[0]), src_ass)].sum()

xx = x.copy()
xx -= xx.min(axis=0, keepdims=True)
xx -= xx.min(axis=1, keepdims=True)

with open('graph', 'w') as f:
    f.write('\n'.join(np.hstack(xx).astype('str')))


