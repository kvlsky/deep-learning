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

# def softmax1(x):
#     # _softmax_out = []
#     softmax_out = np.zeros(x.shape)
#     for map_num in range(x.shape[-1]):
#         for r in np.arange(0, x.shape[0]):
#             for c in np.arange(0, x.shape[1]):
#                 e_x = np.exp([feature_map[r, c, map_num], 0] -
#                  np.max([feature_map[r, c, map_num], 0]))
#                 # e_x = np.exp([feature_map[r, c, map_num], 0] - np.max(
#                 #     [feature_map[r, c, map_num], 0]))
#                 print(e_x / e_x.sum(),r,c,map_num)
#                 softmax_out[r, c, map_num] = e_x / e_x.sum()

#     return softmax_out

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

# def dense()

def backprop(image, filt1, filt2, bias1, bias2, theta3, bias3):
    gradient = []
    dfilt1,  dfilt2,  dbias1,  dbias2,  dtheta3, dbias3 = 0,0,0,0,0,0

    gradient.append(dfilt1)
    gradient.append(dfilt2)
    gradient.append(dbias1)
    gradient.append(dbias2)
    gradient.append(dtheta3)
    gradient.append(dbias3)
    return gradient


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

softmax1 = softmax(max_pool2)
print('\n==========================\nSoftmax\n==========================\n', softmax1)
