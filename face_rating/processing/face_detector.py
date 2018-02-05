#!/usr/bin/env python3

from typing import List
import numpy as np

import cv2

FACE_EXTRA_TOP_PERC = 0.25  # Percentage of extra pixels to include on top of face
WHITE = (255, 255, 255)
PATH_TO_FACEDETECTOR_XML = '/opt/opencv/data/haarcascades/haarcascade_frontalface_default.xml'


def get_resizing_parameters(image_size, target_size):
    """
    Get width, height and top, bottom, left, right border sizes
    """
    # Image size is (height, width)
    image_width = image_size[1]
    image_height = image_size[0]
    dest_width, dest_height = target_size

    # Ratios to get to destination size
    height_ratio = dest_height / image_height
    width_ratio = dest_width / image_width

    # We use the min of those ratios
    ratio = min(height_ratio, width_ratio)

    # We compute the sizes after resizing (and handle casting to integers)
    resized_width = int(round(ratio * image_width))
    resized_height = int(round(ratio * image_height))

    # We then compute the border
    h_border = dest_height - resized_height
    w_border = dest_width - resized_width

    return (
        resized_width,
        resized_height,
        int(h_border / 2),
        h_border - int(h_border / 2),
        int(w_border / 2),
        w_border - int(w_border / 2))


def resize_keep_aspect_ratio(image, target_size=(224, 224)):
    """
    Resize while maintaining aspect ratio. We fill the borders with a constant grey.
    """
    (resized_width,
     resized_height,
     upper_border,
     lower_border,
     left_border, right_border) = get_resizing_parameters( image.shape, target_size)

    image_data = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

    image_data = cv2.copyMakeBorder(
        image_data,
        upper_border,
        lower_border,
        left_border,
        right_border,
        cv2.BORDER_CONSTANT,
        value=WHITE)

    return image_data


def isolate_faces(image: np.array, target_size=(224, 224)) -> List[np.array]:
    """
    Takes an image and returns the faces detected in that image at reshaped to the target_size shape
    :param image:
    :param target_size:
    :return: list of faces
    """
    face_cascade = cv2.CascadeClassifier(PATH_TO_FACEDETECTOR_XML)

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)
    resized_faces = []
    for (x, y, w, h) in faces:
        face_color = image[max(0, y - int(h * FACE_EXTRA_TOP_PERC)):min(y + int((1 + FACE_EXTRA_TOP_PERC) * h),
                                                                        image.shape[0]),
                     x:x + w]
        if target_size is not None:
            face_color = resize_keep_aspect_ratio(face_color, target_size)
        resized_faces.append(face_color)
    return resized_faces
