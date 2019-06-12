
import numpy as np
from random import randint

'''

Aufgabe 1

'''

bitlist = []

for i in range(0,256):
    bitlist.append(i)

def int2bit(x):
    return "{0:b}".format(x)

dicti = {}

for i in range(len(bitlist)):
    key = bitlist[i]
    x = int2bit(key)

    arr = list(x)
    np_arr = np.array(arr)
    np_arr = np_arr.astype(int)
    dicti[key] = np_arr


def generate_random_addition_problem(a,b):

    if a == 255 and b == 255:
        return 0
    else:
        pass

    a_bit = int2bit(a)
    b_bit = int2bit(b)

    c = int(a_bit,2) + int(b_bit,2)
    c = int2bit(c)
    return c

a = randint(0,255)
b = randint(0,255)

out = generate_random_addition_problem(a,b)
print('\n',out)

'''

Aufgabe 2

'''

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_output_to_derivative(x):
    return x * (1 - x)


'''

Aufgabe 3

https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/recurrent_neural_networks.html

'''

def rnn_forward(x, prev_h, Wx, Wh, b):
  xWx = np.dot(x, Wx)
  phWh = np.dot(prev_h,Wh)
  affine = xWx + phWh + b.T
  next_h = sigmoid(affine)

  cache = (x, prev_h.copy(), Wx, Wh, next_h, affine)

  return next_h, cache

def rnn_backward_step(dnext_h, cache):
    (x, prev_h, Wx, Wh, next_h, affine) = cache

    dt = (1 - np.square(sigmoid(affine))) * (dnext_h)

    dxWx = dt
    dphWh = dt
    db = np.sum(dt, axis=0)

    dWh = prev_h.T.dot(dphWh)
    dprev_h = Wh.dot(dphWh.T).T

    dx = dxWx.dot(Wx.T)
    dWx = x.T.dot(dxWx)

    return dx, dprev_h, dWx, dWh, db

def rnn_forward_prop(x, h0, Wx, Wh, b):
  N, T, D = x.shape

  h, cache = None, None
  H = h0.shape[1]
  h = np.zeros((N,T,H))

  h[:,-1,:] = h0
  cache = []

  for t in range(T):
    h[:,t,:], cache_step = rnn_forward(x[:,t,:], h[:,t-1,:], Wx, Wh, b)
    cache.append(cache_step)

  return h, cache

def rnn_backward(dh, cache):
  dx, dh0, dWx, dWh, db = None, None, None, None, None

  N,T,H = dh.shape
  D = cache[0][0].shape[1]


  dWx, dWh, db = np.zeros((D, H)), np.zeros((H, H)), np.zeros((H,))
  dh = dh.copy()

  for t in reversed(range(T)):
    dh[:,t,:]  += dprev_h
    dx_, dprev_h, dWx_, dWh_, db_ = rnn_backward_step(dh[:,t,:], cache[t])
  
    dx[:,t,:] += dx_
    dWx += dWx_
    dWh += dWh_
    db += db_

  dh0 = dprev_h

  return dx, dh0, dWx, dWh, db