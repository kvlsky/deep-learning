from keras.preprocessing.image import (
    ImageDataGenerator,
    array_to_img,
    img_to_array,
    load_img,
)
import numpy as np


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
