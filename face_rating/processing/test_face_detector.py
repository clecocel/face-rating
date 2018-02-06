# Created by: rschucker
# Date: 2/3/18
# ------------------------------

import cv2
from face_rating.processing import face_detector
import matplotlib.pyplot as plt
import os


def plot_image(image):
    plt.figure()
    plt.imshow(image)
    plt.show(block=False)


def main(path='/home/rschucker/Documents/face/data/test_images/860_main_beauty.png'):
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    image = face_detector.reshape_to_max_dimensions(image, max_one_dimension_pixel=800)

    plot_image(image)
    all_faces = face_detector.isolate_faces(image)
    print('found: {} faces'.format(len(all_faces)))
    for face in all_faces:
        plot_image(face)
    plt.show()
    return


def test_large_faces(dir_path='/home/rschucker/Documents/face/data/test_images/large_faces'):
    all_files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    for f in all_files:
        print(f)
        plt.close('all')
        main(os.path.join(dir_path, f))

if __name__ == '__main__':
    main('/home/rschucker/Documents/face/data/test_images/large_faces/16.jpg')
    # test_large_faces()

