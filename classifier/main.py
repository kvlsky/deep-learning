import get_files
import ImgClf
import argparse
import cv2
import numpy as np
from keras import backend as K
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib
import pickle

def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, default_image_size)
            return img_to_array(image)
        else:
            return np.array([])
    except Exception as e:
        print(f"Error occured: {e}")
        return None


def show_plot(history):
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

# Get files
train_files, train_labels = get_files.get_data("classifier\\train.txt")
val_files, val_labels = get_files.get_data("classifier\\val.txt")
test_files = get_files.get_test_data("classifier\\test.txt")

epochs = 100
input_shape = ()

x_train = []
y_train = []

x_test = []
y_test = []

x_val = []
y_val = []

img_rows, img_cols = 28, 28
(x_train, y_train), (x_test, y_test) = (),()

if K.image_data_format() == "channels_first":
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

model = ImgClf.ImgClf(input_shape, epochs)

history = model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=epochs,
    # steps_per_epoch=len(x_train),
    validation_data=(x_test, y_test),
)

show_plot(history)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

print('Saving Model...')
pickle.dump(model, open('cnn_model.pkl', 'wb'))