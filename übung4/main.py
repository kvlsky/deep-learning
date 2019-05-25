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
        idx = np.argpartition(a, -nan_count-1, axis=None)[-nan_count-1]
        multi_idx = np.unravel_index(idx, a.shape)
    return multi_idx


def conv(image, f, bias):
    D, H, W = image.shape
    d, h, w = f.shape

    padding = 0
    stride = 1

    h_out = (H - h + 2 * padding) / stride + 1
    w_out = (W - w + 2 * padding) / stride + 1

    feature_map = np.zeros((D, int(h_out), int(w_out)))

    S = W
    s = w

    for dim in range(D):
        i = 0
        while i < S-1:
            j = 0
            while j < S-1:
                window = image[dim, j:s+j, i:s+i]
                f_map = np.dot(window, f[0])
                f_map = np.dot(f_map, bias)
                f_1 = np.sum(f_map)
                feature_map[dim, int(i/stride), int(j/stride)] = f_1
                j += stride
                pass

            i += stride
            pass

    return feature_map


def relu(feature_map):
    relu_out = np.zeros(feature_map.shape)
    for map_num in range(feature_map.shape[-1]):
        for r in np.arange(0, feature_map.shape[0]):
            for c in np.arange(0, feature_map.shape[1]):
                relu_out[r, c, map_num] = np.max(
                    [feature_map[r, c, map_num], 0])
    return relu_out


def max_pooling(image, f, strd):
    D, H, W = image.shape
    filter_size = f

    stride = strd

    h_out = (H - f) / stride + 1
    w_out = (W - f) / stride + 1

    pool = np.zeros((D, int(h_out), int(w_out)))

    S = W

    for dim in range(D):
        i = 0
        while i < S-1:
            j = 0
            while j < S-1:
                pooling = image[dim, j:filter_size+j, i:filter_size+i]
                p_1 = np.max(pooling)
                pool[dim, int(i/stride), int(j/stride)] = p_1
                j += stride
                pass

            i += stride
            pass
    return pool


X = np.random.rand(3, 6, 6)
l1_filter = np.random.rand(1, 2, 2)
bias = 1

feature_map = conv(X, l1_filter, bias)
print('\n==========================Feature Map==========================\n', feature_map)

feature_map_relu = relu(feature_map)
print('\n==========================Feature Map ReLU==========================\n', feature_map_relu)

max_pool = max_pooling(feature_map_relu, 2, 2)
print('\n==========================Max Pooling==========================\n', max_pool)
