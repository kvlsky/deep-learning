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

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0.0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector

def conv(image, f, bias):

    padding = 0
    stride = 1

    if padding > 0:
        img = np.zeros((image.shape[0], image.shape[1]+padding*2, image.shape[2]+padding*2))
        for dim in range(image.shape[0]):
            img[dim] = np.pad(image[dim], padding, pad_with)
        image = img

    D, H, W = image.shape
    d, h, w = f.shape

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


def softmax(X, theta = 1.0, axis = None):
    y = np.atleast_2d(X)

    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    y = y * float(theta)
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    y = np.exp(y)

    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    p = y / ax_sum

    if len(X.shape) == 1: p = p.flatten()

    return p

def dense(X, weight):
    z = np.zeros(X.shape)
    z2 = np.zeros(X.shape)
    w1 = weight

    for dim in range(X.shape[0]):
        z[dim] = np.dot(X[dim], w1[dim])
        z2[dim] = softmax(z[dim])
        pass

    o = z2
    return o

X = np.random.rand(3, 6, 6)

l1_filter = np.random.rand(1, 2, 2)
bias1 = 1

l2_filter = np.random.rand(1, 2, 2)
bias2 = 1

feature_map = conv(X, l1_filter, bias1)
print('\n==========================\nFeature Map 1\n==========================\n', feature_map)

feature_map_relu = relu(feature_map)
print('\n==========================\nFeature Map ReLU 1\n==========================\n', feature_map_relu)

max_pool = max_pooling(feature_map_relu, 2, 2)
print('\n==========================\nMax Pooling 1\n==========================\n', max_pool)

feature_map2 = conv(feature_map_relu, l2_filter, bias2)
print('\n==========================\nFeature Map 2\n==========================\n', feature_map2)

feature_map_relu2 = relu(feature_map2)
print('\n==========================\nFeature Map ReLU 2\n==========================\n', feature_map_relu2)

max_pool2 = max_pooling(feature_map_relu2, 2, 2)
print('\n==========================\nMax Pooling 2\n==========================\n', max_pool2)

theta3 = np.random.rand(3, 2, 2)
bias3 = 1

dense_out = dense(max_pool2, theta3)
print('\n==========================\nDense\n==========================\n', dense_out)

