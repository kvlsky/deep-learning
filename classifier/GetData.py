import os
from os import listdir
from os.path import isfile, join
import pandas as pd
from PIL import Image
from shutil import copyfile
import os


def get_dataset(path: str):
    files = []
    labels = []
    f = pd.read_csv(path, header=None, delim_whitespace=True)
    for idx, cl in zip(f[0], f[1]):
        files.append(idx)
        labels.append(cl)
    return files, labels


def get_image_dim(img_path: str):
    img = Image.open(img_path)
    width, height = img.size
    return width, height


def get_data():
    # Get files
    train_files, train_labels = get_dataset("classifier\\train.txt")
    val_files, val_labels = get_dataset("classifier\\val.txt")
    test_files, test_labels = get_dataset("classifier\\test.txt")

    source_path = "classifier\\Images\\"
    img_ids = [f for f in listdir(source_path) if isfile(join(source_path, f))]

    w, h = get_image_dim(source_path + img_ids[0])
    image_shape = (3, w, h)

    img_train = []
    img_test = []
    img_val = []

    for img in img_ids:
        if img in train_files:
            img_train.append(source_path + "\\" + img)
        elif img in test_files:
            img_test.append(source_path + "\\" + img)
        elif img in val_files:
            img_test.append(source_path + "\\" + img)
        else:
            pass

    return (
        img_train,
        train_labels,
        img_test,
        test_labels,
        img_val,
        val_labels,
        img_ids,
        image_shape,
    )
