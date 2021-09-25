import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import cv2


def compute_gradient(image):
    Sx = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    Sy = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    Gx = scipy.signal.convolve2d(image, Sx, mode='same')
    Gy = scipy.signal.convolve2d(image, Sy, mode='same')
    im_magnitude = np.sqrt(np.square(Gx) + np.square(Gy))
    im_direction = np.arctan(Gy / Gx)
    return Gx, Gy, im_magnitude, im_direction


if __name__ == '__main__':
    salt_and_pepper = cv2.imread('images/image1_saltpepper.jpg', cv2.IMREAD_GRAYSCALE)
    salt_and_pepper = np.array(salt_and_pepper, dtype='float32')
    salt_and_pepper /= 255.0
    compute_gradient(salt_and_pepper)
