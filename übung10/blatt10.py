# -*- coding: utf-8 -*-
"""
Deep Learning
Übungsblatt 10

@author: Florian
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import cv2


"""
Prepare dataset with train, validation and test data
"""
train_data = pd.read_csv('train.txt', sep=" ", header=None)
train_data.columns = ["feature", "label"]
train_features = train_data["feature"]
train_labels = train_data["label"]

validation_data = pd.read_csv('val.txt', sep=" ", header=None)
validation_data.columns = ["feature", "label"]
validation_features = validation_data["feature"]
validation_labels = validation_data["label"]
for i in range(train_features):
    image = cv2.imread(train_features[i])
    

image_shape = (100,100,3)

"""
Build the keras sequential model
according to http://cs231n.github.io/convolutional-networks/

INPUT -> [CONV -> RELU -> POOL]*2 -> FC -> RELU -> FC
"""
model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=image_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
# calculate output shape of maxpooling layer and use it as input shape of next conv2d layer
maxpool_output_shape=(100,100,1)
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_siize=(2, 2), strides=None, padding='valid', data_format=None))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(102, activation='softmax'))
