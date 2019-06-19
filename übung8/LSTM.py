from keras.models import Sequential
from keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
from random import randint
import generate_data as gend


"""

Aufgabe 7

"""
n_training_samples = 100
n_bit = 8
n_features = 2
n_output = 1

print("\n====================\nAufgabe 7\n====================")

(X, y) = gend.prepareTrainingData(n_training_samples, n_bit, n_features, n_output)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.75, random_state=42
)
print(X.shape, y.shape)

RNN = layers.LSTM
HIDDEN_SIZE = 16
BATCH_SIZE = 512
LAYERS = 1
DIGITS = 8
EPOCHS = 5
INPUT_SHAPE = X.shape[1], X.shape[-1]
OUT_SHAPE = y.shape[-1]

print("Build model...")
model = Sequential()
model.add(
    RNN(
        units=HIDDEN_SIZE,
        activation="sigmoid",
        use_bias=False,
        input_shape=INPUT_SHAPE,
    )
)
model.add(layers.RepeatVector(DIGITS))

for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

model.add(layers.TimeDistributed(layers.Dense(OUT_SHAPE, activation="softmax")))
model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
model.summary()

for iteration in range(EPOCHS):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X_train, y_train,
                batch_size=BATCH_SIZE,
                epochs=1)
    testScore = model.evaluate(X_test, y_test, verbose=0)
    print('Test\nloss - %.6f, accuracy - %.6f' % (testScore[0], testScore[1]))
