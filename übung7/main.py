
import numpy as np
from random import randint

'''

Aufgabe 1

'''
print('\n====================\nAufgabe 1\n====================')

bitlist = []

for i in range(0,256):
    bitlist.append(i)

def int2bit(x):
    return "{0:08b}".format(x)

key_val = {}

for i in range(len(bitlist)):
    key = bitlist[i]
    x = int2bit(key)

    arr = list(x)
    np_arr = np.array(arr)
    np_arr = np_arr.astype(int)
    key_val[key] = np_arr


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
print(out)

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
print('\n====================\nAufgabe 3\n====================')

def forwadStepRNN(X, W_i, h=0, W_h=0, W_o=0):
  xW_i = np.dot(X, W_i)
  hW_h = np.dot(h, W_h)
  input_act = xW_i + hW_h
  next_h = sigmoid(input_act)
  
  cache = (X, h, W_i, W_h, next_h, input_act)
  return next_h, cache


X = np.zeros((len(bitlist),8))

for i in range(X.shape[0]):
      X[i,:] = key_val[i]

W_i = np.random.rand(8,len(X))
W_h = np.random.rand(8,len(X))

next_h, cache = forwadStepRNN(X, W_i)
print('NEXT H\n', next_h)


'''

Aufgabe 4

'''

def mae(y, out):
    return np.sum(np.absolute(y - out))


'''

Aufgabe 5

'''
print('\n====================\nAufgabe 5\n====================')

'''

Aufgabe 6

'''
print('\n====================\nAufgabe 6\n====================')