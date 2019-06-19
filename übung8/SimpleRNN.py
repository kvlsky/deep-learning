from keras.models import Sequential
from keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
from random import randint

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
    X = np.zeros((n_training_samples, n_bit, n_features), dtype=int)
    y = np.zeros((n_training_samples, n_bit, n_output), dtype=int)

    for i in range(n_training_samples):
        (a,b,c,a_bit,b_bit,c_bit) = generate_random_addition_problem()
        for j in range(n_bit):
            X[i,j,0]=a_bit[j]
            X[i,j,1]=b_bit[j]
            y[i,j,0]=c_bit[j]
    return X, y

(X, y)=prepareTrainingData(n_training_samples, n_bit, n_features, n_output)

print ('---------\nX:\n-----------\n', X)
print ('---------\ny:\n-----------\n', y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=42)



RNN = layers.SimpleRNN
HIDDEN_SIZE = 16
BATCH_SIZE = 128
LAYERS = 1
MAXLEN = 8
DIGITS = 8
EPOCHS = 10
INPUT_SIZE = 0

print('Build model...')
model = Sequential()
model.add(RNN(units=HIDDEN_SIZE,activation='sigmoid', use_bias=False))
model.add(layers.RepeatVector(DIGITS))

for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

model.add(layers.TimeDistributed(layers.Dense(INPUT_SIZE, activation='softmax')))
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

for iteration in range(1, 200):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X_train, y_train,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS)

    trainPredict = model.predict(X_train)
    testPredict= model.predict(X_test)
    predicted=np.concatenate((trainPredict,testPredict),axis=0)

    trainScore = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: ', trainScore)
    testScore = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score: ', testScore)
