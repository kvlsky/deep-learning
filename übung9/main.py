from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import random
import sys

'''

Aufgabe 1: Daten Präparation

'''

with open('übung9\\nietzsche.txt', encoding='utf-8') as f:
    text = f.read().lower()
print(f'lenght of the corpus - {len(text)}')

chars = sorted(list(set(text)))
print(f'total number of chars - {len(chars)}')

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 40
# step = 3
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print(f'nb sequences - {len(sentences)}')

print('Vectorization...')

x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


'''

Aufgabe 2:  LSTM Modell

'''


print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.summary()

'''

Aufgabe 3:  Sample Funktion

'''

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

'''

Aufgabe 4:  Testausgabe

'''


def on_epoch_end(epoch, _):
    print('\n')
    print(f'----- Generating text after Epoch: {epoch}')

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print(f'----- diversity: {diversity}')

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print(f'----- Generating with seed: "{sentence}"')
        sys.stdout.write(generated)

        for _ in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)
            preds = preds[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print('\n')

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y,
          batch_size=128,
          epochs=1,
          callbacks=[print_callback])