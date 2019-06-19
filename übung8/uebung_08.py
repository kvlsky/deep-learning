# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:32:03 2019

@author: Florian
"""


import numpy as np
from random import randint
from keras.models import Sequential
from keras import layers

'''

Aufgabe 7

'''
print('\n====================\nAufgabe 7\n====================')

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

# Prepare the training data
n_training_samples = 100
n_bit = 8
n_features = 2
n_output = 1

def prepareTrainingData (n_training_samples, n_bit, n_features, n_output):
    Train_X = np.zeros((n_training_samples, n_bit, n_features), dtype=int)
    Train_Y = np.zeros((n_training_samples, n_bit, n_output), dtype=int)

    for i in range(n_training_samples):
        (a,b,c,a_bit,b_bit,c_bit) = generate_random_addition_problem()
        for j in range(n_bit):
            Train_X[i,j,0]=a_bit[j]
            Train_X[i,j,1]=b_bit[j]
            Train_Y[i,j,0]=c_bit[j]
    return Train_X, Train_Y

(Train_X, Train_Y)=prepareTrainingData(n_training_samples, n_bit, n_features, n_output)

print ('---------\nTrain_X:\n-----------\n', Train_X)
print ('---------\nTrain_Y:\n-----------\n', Train_Y)


'''

Aufgabe 8

'''
print('\n====================\nAufgabe 8\n====================')
print('Build model...')
model=Sequential()
RNN = layers.SimpleRNN
HIDDEN_SIZE = 16
BATCH_SIZE = 128
LAYERS = 1

model.add(RNN(HIDDEN_SIZE, input_shape=(1,2)))
model.add(layers.RepeatVector(n_bit))

for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

model.add(layers.TimeDistributed(layers.Dense(n_features, activation='sigmoid')))
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

for iteration in range(0, n_training_samples):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    for i in range(n_bit):
        model.fit(Train_X[iteration, i], Train_Y[iteration, i],
                  batch_size=BATCH_SIZE,
                  epochs=1)
#    model.fit(Train_X[iteration], Train_Y[iteration],
#              batch_size=BATCH_SIZE,
#              epochs=1 ) #,
              # validation_data=(x_val, y_val))

preds = model.predict([0,1,1,0,1,0,0,0],[0,0,0,0,1,1,1,1])

print('---------\nPrediction:\n----------', preds)
