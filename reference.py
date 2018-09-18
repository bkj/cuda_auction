#!/usr/bin/env python

"""
    reference.py
"""

import sys
import numpy as np
from time import time
from lapjv import lapjv

data = [int(x) for x in open('graph').read().splitlines()]
num_nodes = np.sqrt(len(data)).astype(int)

data = np.array(data).reshape(num_nodes, num_nodes)

t = time()
src_ass, _, _ = lapjv(data.max() - data)
elapsed = 1000 * (time() - t)
score = int(data[(np.arange(num_nodes), src_ass)].sum())

print('score=%d' % score, file=sys.stderr)
print('time=%f' % elapsed, file=sys.stderr)