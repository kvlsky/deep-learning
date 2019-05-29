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

def softmax(feature_map):
    softmax_out = np.zeros(feature_map.shape)

    for dim in range(feature_map.shape[0]):
        x = feature_map[dim]
        e_x = np.exp(x - np.max(x))
        res = e_x / e_x.sum()
        softmax_out[dim] = res

    return softmax_out

class FFNN(object):
    def __init__(self, weights):
        self.inputSize = 3
        self.outputSize = 1
        self.hiddenSize = 1
        self.w1 = weights
        self.w2 = np.random.randn(self.hiddenSize,self.outputSize)

    def forward(self, X):
        # dot product of input and first set of weights
        print(self.w1.shape, X.shape)
        self.z = np.dot(X,self.w1)
        # activation function 
        self.z2 = softmax(self.z)  
        return self.z2

    def softmaxDer(self, x):
        s = x.reshape(-1,1)
        softDer = np.diagflat(s) - np.dot(s, s.T)
        return softDer

    def backward(self, X, y, o):
        # backward propgate through the network
        self.o_error = y - o 
        self.o_delta = self.o_error * self.softmaxDer(o)

        self.z2_error = self.o_delta.dot(self.w2.T)
        self.z2_delta = self.z2_error * self.softmaxDer(self.z2)

        # adjusting weights
        self.w1 += X.T.dot(self.z2_delta)
        self.w2 += self.z2.T.dot(self.o_delta)

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

    def predict(self):
        print( "Predicted data based on trained weights: ")

