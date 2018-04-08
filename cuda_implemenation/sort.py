#!/usr/bin/env python3

import sys
import numpy as np

with open (sys.argv[1]) as f:
    arr = np.array([ np.float(line.split(sep=' ')[1]) for line in f.readlines()])
    print (arr)
    print (np.argsort(arr))