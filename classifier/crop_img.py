import os
from os import listdir
from os.path import isfile, join
import pandas as pd
from PIL import Image
from shutil import copyfile


def crop_image(size, img_id, img_path: str):
    img = Image.open(img_path)
    img.thumbnail(size, Image.ANTIALIAS)

    path = 'classifier\\data\\'

    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass

    print(f'Saving cropped img to {path}\\{img_id}')
    img.save(path + img_id, 'JPEG', quality=200)

source_path = "classifier\\Images\\"
img_ids = [f for f in listdir(source_path) if isfile(join(source_path, f))]

size = 360, 360
for img in img_ids[:2]:
    print(f'Cropping image {img}')
    crop_image(size, img, source_path + img)
