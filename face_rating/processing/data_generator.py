#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.model_selection import train_test_split

FACE_RATING_HOME = '/home/ubuntu/dataset/SCUT-FBP5500'  # os.environ.get('FACE_RATING_HOME')

DATA_DIR = 'data'
IMAGES_PATH = os.path.join(FACE_RATING_HOME, DATA_DIR, 'Images')
RATINGS_PATH = os.path.join(FACE_RATING_HOME, DATA_DIR, 'ratings_mean.csv')
IMAGE_COL = 'Image'
RATING_COL = 'Rating'

RANDOM_STATE = 0
MAX_RGB_VALUE = 255


def load_image(path, target_size=(224, 224)):
    img = load_img(path, target_size=target_size, interpolation='bicubic')
    x = img_to_array(img)
    return x


def load_x_y(data_path, rating_path, target_size=(224, 224)):
    all_jpg_files = [f for f in os.listdir(data_path) if f.endswith('.jpg')]
    scores_s = load_scores(rating_path)
    scores_list = [scores_s[f] for f in all_jpg_files]
    images = [load_image(os.path.join(data_path, f), target_size=target_size) for f in all_jpg_files]

    return np.array(images), np.array(scores_list)


def load_scores(rating_path):
    df = pd.read_csv(rating_path)
    df.index = df[IMAGE_COL]
    series = df[RATING_COL]
    return series


def main(target_size=(224, 224), test_split=0.20, batch_size=32, data_augmentation=True):
    x, y = load_x_y(IMAGES_PATH, RATINGS_PATH, target_size=target_size)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split, random_state=RANDOM_STATE)
    if data_augmentation:
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            rescale=1/MAX_RGB_VALUE,
        )
    else:
        datagen = ImageDataGenerator(rescale=1/MAX_RGB_VALUE)

    gen = datagen.flow(x_train, y_train, batch_size=batch_size)
    return gen, x_train.shape[0], (x_test, y_test), x_test.shape[0]


if __name__ == '__main__':
    main()
