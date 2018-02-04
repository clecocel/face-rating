# !/usr/bin/env python3

import logging as logs
import inspect
import numpy as np

from keras.models import Model
import cv2

from face_rating.processing import face_detector


def main(model: Model, image: np.array, input_size=(224, 224)):
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
    x = np.array(faces)
    model.predict(x)


if __name__ == '__main__':
    path = '/home/rschucker/Documents/face/data/test_images/860_main_beauty.png'
    image = cv2.imread(path)
    main(None, image)

