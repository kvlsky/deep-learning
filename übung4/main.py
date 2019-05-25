import numpy as np
import gzip
import scipy.io as sio
import pickle
import time
import random

def nanargmx(a):
    idx = np.argmax(a, axis=None)
    multi_idx = np.unravel_index(idx, a.shape)
    if np.isnan(a[multi_idx]):
        nan_count = np.sum(np.isnan(a))
        idx = np.argpartition(a,-nan_count-1, axis=None)[-nan_count-1]
        multi_idx = np.unravel_index(idx, a.shape)
    return multi_idx

def maxpool(X, filter_size, stride):
    (l,w,w) = X.shape
    pool = np.zeros((l, int((w-filter_size)/stride + 1),int((w-filter_size)/stride + 1)))
    
    for jj in range(0,l):
        i=0
        while i < w:
            j=0
            while j < w:
                pool[j, int(i/2), int(j/2)] = np.max(X[jj, i:i+filter_size, j:j+filter_size])
                j += stride
            i += stride
    return pool


X = np.random.randint(0, 100, size=(5, 5, 2))

print(maxpool(X, 2, 1))
