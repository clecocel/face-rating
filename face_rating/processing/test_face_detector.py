# Created by: rschucker
# Date: 2/3/18
# ------------------------------

import cv2
from face_rating.processing import face_detector
import matplotlib.pyplot as plt


def plot_image(image):
    plt.figure()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show(block=False)


def main():
    path = '/home/rschucker/Documents/face/data/test_images/860_main_beauty.png'
    image = cv2.imread(path)
    plot_image(image)
    all_faces = face_detector.isolate_faces(image)
    for face in all_faces:
        plot_image(face)
    plt.show()
    return

if __name__ == '__main__':
    main()

