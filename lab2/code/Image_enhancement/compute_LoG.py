import cv2
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import math


def mask(n):
    mask = np.ones((n, n))
    mask[n // 2][n // 2] = 1 - n ** 2
    return mask


def compute_LoG(image, **kwargs):
    filter = np.zeros(kwargs['filter_size'])
    xl, yl = kwargs['filter_size']
    xl = int(xl / 2)
    yl = int(yl / 2)
    if kwargs['LOG_type'] == 1:
        d = kwargs['sigma']
        for x in np.arange(-xl, xl + 1, 1):
            for y in np.arange(-yl, yl + 1, 1):
                ans = 1 / (2 * np.pi * d ** 2)
                ans *= np.exp(-(x ** 2 + y ** 2) / (2 * d ** 2))
                filter[y + yl][x + xl] = ans
        smoothed = scipy.signal.convolve2d(image, filter)
        laplacian = mask(5)
        return scipy.signal.convolve2d(smoothed, laplacian)

    elif kwargs['LOG_type'] == 2:
        d = kwargs['sigma']
        for x in np.arange(-xl, xl + 1, 1):
            for y in np.arange(-yl, yl + 1, 1):
                ans = -1 / (np.pi * d ** 4)
                ans *= (1 - (x ** 2 + y ** 2) / (2 * d ** 2))
                ans *= np.exp(-(x ** 2 + y ** 2) / (2 * d ** 2))
                filter[y + yl][x + xl] = ans
        return scipy.signal.convolve2d(image, filter)

    elif kwargs['LOG_type'] == 3:
        d1 = kwargs['sigma_one']
        d2 = kwargs['sigma_two']
        for x in np.arange(-xl, xl + 1, 1):
            for y in np.arange(-yl, yl + 1, 1):
                ans = 1 / np.sqrt(2 * np.pi)
                ans *= np.exp(-x ** 2 / (2 * d1 ** 2)) - np.exp(-y ** 2 / (2 * d2 ** 2))
                filter[y + yl][x + xl] = ans
        return scipy.signal.convolve2d(image, filter)


if __name__ == '__main__':
    image_2 = cv2.imread('images/image2.jpg', cv2.IMREAD_GRAYSCALE)
    image_2 = np.array(image_2, dtype='float32')
    for i in range(2):
        for j in range(2):
            plt.imshow(compute_LoG(image_2, **{'filter_size': (3, 3), 'LOG_type': 3, 'sigma_one': 0.5, 'sigma_two': 1, 'sigma': 0.5}),
                       cmap='gray')
            plt.show()
