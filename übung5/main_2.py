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

def TwoLayerCNN(image, filt1, filt2, bias1, bias2, theta3, bias3):
    feature_map = conv(image, filt1, bias1)
    print('\n==========================\nFeature Map 1\n==========================\n', feature_map)

    feature_map_relu = relu(feature_map)
    print('\n==========================\nFeature Map ReLU 1\n==========================\n', feature_map_relu)

    max_pool = max_pooling(feature_map_relu, 2, 2)
    print('\n==========================\nMax Pooling 1\n==========================\n', max_pool)


    
    feature_map2 = conv(feature_map_relu, filt2, bias2)
    print('\n==========================\nFeature Map 2\n==========================\n', feature_map2)

    feature_map_relu2 = relu(feature_map2)
    print('\n==========================\nFeature Map ReLU 2\n==========================\n', feature_map_relu2)

    max_pool2 = max_pooling(feature_map_relu2, 2, 2)
    print('\n==========================\nMax Pooling 2\n==========================\n', max_pool2)

    #softmax1 = softmax(max_pool2)
    #print('\n==========================\nSoftmax\n==========================\n', softmax1)
    """
    feature_map_1 = conv(image, filt1, bias1)
    feature_map_relu_1 = relu(feature_map_1)
    max_pool_1 = max_pooling(feature_map_relu_1, 2, 2)
    feature_map_2 = conv(max_pool_1, filt2, bias2)
    feature_map_relu_2 = relu(feature_map_2)
    max_pool_2 = max_pooling(feature_map_relu_2, 2, 2)
    
    """
    dim, hei, wid = max_pool2.shape
    theta3=np.random.rand(dim, hei, wid)
    DenseNetwork = FFNN(theta3)
    output=DenseNetwork.forward(max_pool2)
    return output
    
    # TODO: Dense Layer mit Gewichtsmatrix theta3 und bias bias3
    # mit softmax Aktivierungsfunktion
    
    
    
    

class FFNN(object):
    def __init__(self, weights):
        self.inputSize = 3
        self.outputSize = 3
        self.hiddenSize = 1
        self.w1 = weights 

    def forward(self, X):
        # dot product of input and first set of weights
        self.z = np.dot(X,self.w1)
        # activation function 
        self.z2 = self.softmax1(self.z)  
        return self.z2
    
    
    def softmax1(self,feature_map):
        softmax_out = np.zeros(feature_map.shape)
        for dim in range(feature_map.shape[0]):
            x = feature_map[dim]
            e_x = np.exp(x - np.max(x))
            res = e_x / e_x.sum()
            softmax_out[dim] = res
        return softmax_out
    
    def softmax_2(self, w):
        w = np.array(w)
        softmax_out = np.zeros(w.shape)
        for map_num in range(w.shape[-1]):
            for r in np.arange(0, w.shape[0]):
                for c in np.arange(0, w.shape[1]):
                    npa = np.array
                    t=1
                    e = np.exp(npa(w) / t)
                    dist = e / np.sum(e)
                    softmax_out[r, c, map_num] = dist
        return softmax_out


    def softmax(X, theta = 1.0, axis = None):
        """
        Compute the softmax of each element along an axis of X.

        Parameters
        ----------
        X: ND-Array. Probably should be floats.
        theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
        axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

        Returns an array the same size as X. The result will sum to 1
        along the specified axis.
        """
        # make X at least 2d
        y = np.atleast_2d(X)

        # find axis
        if axis is None:
            axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

        # multiply y against the theta parameter,
        y = y * float(theta)

        # subtract the max for numerical stability
        y = y - np.expand_dims(np.max(y, axis = axis), axis)

        # exponentiate y
        y = np.exp(y)

        # take the sum along the specified axis
        ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

        # finally: divide elementwise
        p = y / ax_sum

        # flatten if X was 1D
        if len(X.shape) == 1: p = p.flatten()

        return p

    def sigmoidDer(self, x):
        #derivative of sigmoid
        return x * (1 - x)

    def backward(self, X, y, o):
        # backward propgate through the network
        self.o_error = y - o 
        self.o_delta = self.o_error * self.sigmoidDer(o)

        self.z2_error = self.o_delta.dot(self.w2.T)
        self.z2_delta = self.z2_error * self.sigmoidDer(self.z2)

        # adjusting weights
        self.w1 += X.T.dot(self.z2_delta)
        self.w2 += self.z2.T.dot(self.o_delta)

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

    def predict(self):
        print( "Predicted data based on trained weights: ")
        #print( "Input (scaled): \n" + str(xPredicted))
        #print( "Output: \n" + str(self.forward(xPredicted)))

    


image = np.random.rand(3, 6, 6)
l1_filter = np.random.rand(1, 2, 2)
l2_filter = np.random.rand(1,2,2)
bias1 = 1
bias2=1
theta3=l1_filter ## todo:entferen
bias3=1

output=TwoLayerCNN(image, l1_filter, l2_filter, bias1, bias2, theta3, bias3)
print('Output: \n',output)
"""
print('\n==========================Input Image==========================\n', X)

feature_map = conv(X, l1_filter, bias)
print('\n==========================Feature Map==========================\n', feature_map)

feature_map_relu = relu(feature_map)
print('\n==========================Feature Map ReLU==========================\n', feature_map_relu)

max_pool = max_pooling(feature_map_relu, 2, 2)
print('\n==========================Max Pooling==========================\n', max_pool)
"""