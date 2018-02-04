# Created by: rschucker
# Date: 2/3/18
# ------------------------------

import logging as logs
import inspect
import numpy as np

from keras.models import Model
import cv2

from face_rating.processing import face_detector


def main(model: Model, image: np.array, input_size=(224, 224)):
    logger = logs.getLogger('{}.{}()'.format(__name__, inspect.currentframe().f_code.co_name))
    logger.debug('Detecting faces in image')
    faces = face_detector.isolate_faces(image, target_size=input_size)
    logger.debug('Found {} faces in image'.format(len(faces)))
    x = np.array(faces)
    model.predict(x)


if __name__ == '__main__':
    path = '/home/rschucker/Documents/face/data/test_images/860_main_beauty.png'
    image = cv2.imread(path)
    main(None, image)

