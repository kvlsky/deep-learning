from __future__ import print_function
from keras import backend as K
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import (
    ImageDataGenerator,
    array_to_img,
    img_to_array,
    load_img,
)
import matplotlib.pyplot as plt
import pickle
import numpy as np

from GetData import get_data
import ImgClf

import argparse


def convert_image_to_array(image_path):
    try:
        image_file = load_img(image_path)
        if image_file is not None:
            image = img_to_array(image_file)
            image = image.reshape((1,) + image.shape)
            return image
        else:
            return np.array([])
    except Exception as e:
        print(f"Error occured: {e}")
        return None


def show_plot(history, epochs):
    acc = history["acc"]
    val_acc = history["val_acc"]
    loss = history["loss"]
    val_loss = history["val_loss"]
    epochs = epochs

    plt.plot(epochs, acc, "b", label="Training accuracy")
    plt.plot(epochs, val_acc, "r", label="Validation accuracy")
    plt.title("Training and Validation accuracy")
    plt.legend()

    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title("Training and Validation loss")
    plt.legend()

    plt.show()


'''

Get all necessary training data

    data = (
        img_train,
        train_labels,
        img_test,
        test_labels,
        img_val,
        val_labels,
        img_ids,
        image_shape,
    )

'''

data = get_data()

epochs = 100
input_shape = data[-1]
num_classes = len(set(data[1]))

x_train = []
y_train = []

x_test = []
y_test = []

x_val = []
y_val = []

# Convert images --> arr and get their labels
for img, label in zip(data[0], data[1]):
    x_train.append(convert_image_to_array(img))
    y_train.append(label)


for img, label in zip(data[2], data[3]):
    x_test.append(convert_image_to_array(img))
    y_test.append(label)

for img, label in zip(data[4], data[5]):
    x_val.append(convert_image_to_array(img))
    y_val.append(label)

model = ImgClf.ImgClf(input_shape, num_classes, epochs)

history = model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=epochs,
    validation_data=(x_test, y_test),
)

show_plot(history)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

print('Saving Model...')
pickle.dump(model, open('cnn_model.pkl', 'wb'))
