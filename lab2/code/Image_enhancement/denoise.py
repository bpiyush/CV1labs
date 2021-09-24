import cv2
import numpy as np
import matplotlib.pyplot as plt


def denoise(image, kernel_type, k):
    if kernel_type == 'box':
        out = cv2.blur(image, (k, k))
    elif kernel_type == 'median':
        out = cv2.medianBlur(image, (k, k))
    elif kernel_type == 'gaussian':
        out = cv2.GaussianBlur(image, (k, k))
    else:
        print('Operation not implemented')
    return out


if __name__ == '__main__':
    salt_and_pepper = cv2.imread('images/image1_saltpepper.jpg', cv2.IMREAD_GRAYSCALE)
    image_one = cv2.imread('images/image1.jpg', cv2.IMREAD_GRAYSCALE)
    gaussian = cv2.imread('images/image1_gaussian.jpg', cv2.IMREAD_GRAYSCALE)
    salt_and_pepper = np.array(salt_and_pepper, dtype='float32')
    image_one = np.array(image_one, dtype='float32')
    gaussian = np.array(gaussian, dtype='float32')
    for k in [3, 5, 7]:
        plt.imshow(denoise(image_one, 'box', k))
        plt.show()
