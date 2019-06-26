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

TEXT_LENGHT = len(text)
print(f'Lenght of the corpus - {TEXT_LENGHT}')

chars = sorted(list(set(text)))
CHARS_LENGHT = len(chars)

print(f'Total number of chars - {CHARS_LENGHT}')

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

MAXLEN = 40
sentences = []
next_chars = []

for i in range(0, TEXT_LENGHT - MAXLEN):
    sentences.append(text[i: i + MAXLEN])
    next_chars.append(text[i + MAXLEN])
print(f'Number of sequences - {len(sentences)}')

print('\n===============Vectorization...===============')

x = np.zeros((len(sentences), MAXLEN, CHARS_LENGHT), dtype=np.bool)
y = np.zeros((len(sentences), CHARS_LENGHT), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


'''

Aufgabe 2:  LSTM Modell

'''


print('\n===============Creating Model...===============')
model = Sequential()
model.add(LSTM(128, input_shape=(MAXLEN, CHARS_LENGHT)))
model.add(Dense(CHARS_LENGHT, activation='softmax'))

opt = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt)
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
EPOCHS = 10

def on_epoch_end(epoch, _):
    print('\n')
    print(f'----- Generating text after Epoch: {epoch}')

    start_index = random.randint(0, TEXT_LENGHT - MAXLEN - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print(f'----- diversity: {diversity}')

        generated = ''
        sentence = text[start_index: start_index + MAXLEN]
        generated += sentence
        print(f'----- Generating with seed: "{sentence}"')
        sys.stdout.write(generated)

        for _ in range(400):
            x_pred = np.zeros((1, MAXLEN, CHARS_LENGHT))
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
          epochs=EPOCHS,
          callbacks=[print_callback])