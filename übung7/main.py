
import numpy as np
from random import randint

'''

Aufgabe 1

'''
print('\n====================\nAufgabe 1\n====================')

bitlist = []

for i in range(0,255):
    bitlist.append(i)

def int2bit(x):
    return "{0:08b}".format(x)

key_val = {}
# create dictionary: key = decimal int, value = bit array
for i in range(len(bitlist)):
    key = bitlist[i]
    x = int2bit(key)

    arr = list(x)
    np_arr = np.array(arr)
    np_arr = np_arr.astype(int)
    key_val[key] = np_arr


def generate_random_addition_problem():
    # Generate random int numbers with maximum sum = 255
    while 1:
        a = randint(0,255)
        b = randint(0,255)
        if (a+b)<=255:
            break
    # TODO: return values as bit integer or bit array?    
    a_bit = key_val[a]
    b_bit = key_val[b]

    c = a + b
    c_bit = key_val[c]
    return a,b,c,a_bit,b_bit,c_bit


out = generate_random_addition_problem()
print(out)

'''

Aufgabe 2

'''

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return np.exp(-x) / ((1+np.exp(-x))**2)

# use only sigmoid(x) instead of x
def sigmoid_output_to_derivative(x):
    return sigmoid(x) * (1 - (sigmoid(x)))


'''

Aufgabe 3

https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/recurrent_neural_networks.html

'''
print('\n====================\nAufgabe 3\n====================')

def forward_RNN_one_step(X, W_i, h=0, W_h=0, W_o=0):
  xW_i = np.dot(X, W_i)
  hW_h = np.dot(h, W_h)
  input_act = xW_i + hW_h
  next_h = sigmoid(input_act)
  o = np.dot(next_h, W_o)
  cache = (X, h, W_i, W_h, next_h, input_act)
  return next_h, o, cache


X = np.zeros((len(bitlist),8))

for i in range(X.shape[0]):
      X[i,:] = key_val[i]

W_i = np.random.rand(8,len(X))
W_h = np.random.rand(8,len(X))
W_o = np.random.rand(8,len(X))

next_h, o, cache = forward_RNN_one_step(X, W_i, 0, W_h, W_o)
print('NEXT H:\n', next_h)
print('OUTPUT:\n', o)


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