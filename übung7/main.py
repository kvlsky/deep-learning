
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
print('Addition problem:\n',out)

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
  return next_h, o , cache


'''

Aufgabe 4


'''
# Mean absolute error
def mae(y, out):
    return np.sum(np.absolute(y - out))


'''

Aufgabe 5

https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/recurrent_neural_networks.html

'''
print('\n====================\nAufgabe 5\n====================')

def backprop_one_step(X, h, h_prev, d_out, d_h_future, W_i, W_h, W_o, input_act):

    # dt: delta of total
    # Gradient of sigmoid times d_h_future
    dt = sigmoid_output_to_derivative(input_act) * (d_h_future)
    print ('dt: ', dt)
    # Gradient of sum block (split input_act)
    dxWi = dt
    dhWh = dt

    # Gradient of the hidden weights matrix W_h
    dWh = h_prev.T.dot(dhWh)
    dprev_h = W_h.dot(dhWh.T).T

    # Gradient of the input weights matrix W_i
    #di = dxWi.dot(W_i.T)
    dWi = X.T.dot(dxWi)
    
    # Gradient of the output weights matrix W_o
    dWo = h.T.dot(d_out)

    return dprev_h, dWi, dWh, dWo
    

'''

Aufgabe 6

'''
print('\n====================\nAufgabe 6\n====================')

# input size: 2 values (one bit of a, one bit of b)
X = np.zeros((1,2), dtype=int)
(a, b, c, a_bit, b_bit, c_bit) = generate_random_addition_problem()
X[0,0]=a_bit[0]
X[0,1]=b_bit[1]

# hidden size = 16
h = np.zeros((1,16), dtype=float)
# weight matrices: output size = 1
W_i = np.random.rand(2,16)
W_h = np.random.rand(16,16)
W_o = np.random.rand(16,1)

def train (X, W_i, h, W_h, W_o):
    next_h, o, cache = forward_RNN_one_step(X, W_i, h, W_h, W_o)
    d_out = mae(c_bit[0], o)
    dprev_h, dWi, dWh, dWo = backprop_one_step(X, next_h, cache[1], d_out, 1, W_i, W_h, W_o, cache[5])
    print('NEXT H:\n', next_h)
    print('OUTPUT:\n', o)
    print('dprev_h:\n',dprev_h)
    print('dWi:\n', dWi)
    print('dWh:\n', dWh)
    print('dWo:\n', dWo)
    
train(X, W_i, h, W_h, W_o)
    





