#!/usr/bin/env python

"""
    make-data.py
"""

import numpy as np
from time import time
from lapjv import lapjv as jv_src

np.random.seed(123)

N = 8192
max_value = 400
x = np.random.choice(max_value, (N, N))

t = time()
src_ass, trg_ass, _ =  jv_src(x.max() - x)
src_ass[:10]
milliseconds = int(1000 * (time() - t))
print(milliseconds)

x[(np.arange(x.shape[0]), src_ass)].sum()

with open('graph', 'w') as f:
    f.write('\n'.join(np.hstack(x).astype('str')))
