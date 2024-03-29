import numpy as np
from random import randint

bitlist = []

for i in range(0, 256):
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
    while True:
        a = randint(0, 255)
        b = randint(0, 255)
        if (a + b) <= 255:
            break
        else:
            pass
    # TODO: return values as bit integer or bit array?
    a_bit = key_val[a]
    b_bit = key_val[b]

    c = a + b
    c_bit = key_val[c]
    return a, b, c, a_bit, b_bit, c_bit


def prepareTrainingData(n_training_samples, n_bit, n_features, n_output):
    X = np.zeros((n_training_samples, n_bit, n_features), dtype=int)
    y = np.zeros((n_training_samples, n_bit, n_output), dtype=int)

    for i in range(n_training_samples):
        (a, b, c, a_bit, b_bit, c_bit) = generate_random_addition_problem()
        for j in range(n_bit):
            X[i, j, 0] = a_bit[j]
            X[i, j, 1] = b_bit[j]
            y[i, j, 0] = c_bit[j]
    return X, y