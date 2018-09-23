#!/usr/bin/env python

"""
    make-data.py
"""

import numpy as np
from time import time
from lap import lapjv

np.random.seed(123)
_ = np.random.choice(10, (10, 10)).astype('float32')

N = 2 ** 12
print('dim', N)
max_value = 100
x = np.random.choice(max_value, (N, N))

t = time()
_, src_ass, _ =  lapjv((x.max() - x))
milliseconds = int(1000 * (time() - t))
print('milliseconds', milliseconds)

print('score', x[(np.arange(x.shape[0]), src_ass)].sum())

with open('graph', 'w') as f:
    f.write('\n'.join(np.hstack(x).astype('str')))


