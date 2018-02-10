# !/usr/bin/env python3

import logging as logs
import inspect
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model, load_model
import cv2

from face_rating.processing import face_detector
from face_rating.processing.data_generator import load_x_y, RATINGS_PATH, IMAGES_PATH


def rank_faces_in_image(model: Model, image: np.array, input_size=(224, 224)):
    """
    Given a model an an image, return the scores of all faces detect along side with the detected face
    :param model: the trained model
    :param image: image to rate
    :param input_size: target in put size for the model
    :return: list of (image, score attributed to that image)
    """
    logger = logs.getLogger('{}.{}()'.format(__name__, inspect.currentframe().f_code.co_name))
    logger.debug('Detecting faces in image')
    faces = face_detector.isolate_faces(image, target_size=input_size)
    logger.debug('Found {} faces in image'.format(len(faces)))
    if len(faces) == 0:
        logger.error('No face detected in this image')
        return
    x = np.array(faces)
    y_pred = model.predict(x)
    y_pred = y_pred.flatten().tolist()
    polarized_scores = [polarize(score) for score in y_pred]
    return faces, polarized_scores

def polarize(score):
    ''' Polarize the score for more fun :) '''
    return np.clip(2.5 + (score - 2.58) * 3, 0, 5)


def plot_scores_of_test_image(image_path, model_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    model = load_model(model_path)
    faces, y_pred = rank_faces_in_image(model, image)

    for f, y in zip(faces, y_pred):
        plt.figure()
        plt.imshow(f)
        plt.title('Score {:2.1f}/5'.format(y))
        plt.show(block=False)
    plt.show()


def plot_predicted_scores_of_data_set():
    logger = logs.getLogger('Plot all')
    logger.info('Loading images...')
    x, y = load_x_y(IMAGES_PATH, RATINGS_PATH)
    model_path = '/home/rschucker/Documents/face/data/trained_model/resnet-trained.h5'
    logger.info('Loading model...')
    model = load_model(model_path)
    # step = 550
    step = 1
    x_splits = np.split(x, step)
    y_splits = np.split(y, step)
    for x_split, y_split in zip(x_splits, y_splits):
        y_pred = model.predict(x_split)
        y_pred = y_pred.flatten().tolist()
        for img, true_score, model_score in zip(x_split, y_split, y_pred):
            plt.figure()
            plt.imshow(np.uint8(img))
            plt.title('True Score {:2.1f}/5, Model Score {:2.1f}/5'.format(true_score, model_score))
            plt.show(block=False)
        plt.show()
        plt.close('all')


def main():
    # image_path = '/home/rschucker/Documents/face/data/test_images/860_main_beauty.png'
    image_path = '/home/rschucker/Documents/face/data/Images/fty1357.jpg'
    model_path = '/home/rschucker/Documents/face/data/trained_model/resnet-trained.h5'
    plot_scores_of_test_image(image_path, model_path)


if __name__ == '__main__':
    root_logger = logs.getLogger()
    logs.basicConfig(level=logs.INFO)
    root_logger.setLevel(logs.INFO)

    main()
    # plot_predicted_scores_of_data_set()
