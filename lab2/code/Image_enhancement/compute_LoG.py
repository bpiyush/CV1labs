import cv2
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import math


def mask(n):
    mask = np.ones((n, n))
    mask[n // 2][n // 2] = 1 - n ** 2
    return mask


def compute_LoG(image, LOG_type, d1=0, d2=0):
    if LOG_type == 1:
        k = 5
        gauss = np.zeros((k, k))
        d = 0.5
        b = int(k / 2)
        for x in np.arange(-b, b + 1, 1):
            for y in np.arange(-b, b + 1, 1):
                ans = 1 / (2 * np.pi * d ** 2)
                ans *= np.exp(-(x ** 2 + y ** 2) / (2 * d ** 2))
                gauss[y + b][x + b] = ans
        smoothed = scipy.signal.convolve2d(image, gauss)
        laplacian = mask(k)
        return scipy.signal.convolve2d(smoothed, laplacian)

    elif LOG_type == 2:
        k = 5
        LoG = np.zeros((k, k))
        d = 0.5
        b = int(k / 2)

        for x in np.arange(-b, b + 1, 1):
            for y in np.arange(-b, b + 1, 1):
                ans = -1 / (np.pi * d ** 4)
                ans *= (1 - (x ** 2 + y ** 2) / (2 * d ** 2))
                ans *= np.exp(-(x ** 2 + y ** 2) / (2 * d ** 2))
                LoG[y + b][x + b] = ans
        return scipy.signal.convolve2d(image, LoG)

    elif LOG_type == 3:
        k = 5
        DoG = np.zeros((5, 5))
        b = int(k / 2)
        for x in np.arange(-b, b + 1, 1):
            for y in np.arange(-b, b + 1, 1):
                ans = 1 / np.sqrt(2 * np.pi)
                ans *= np.exp(-x ** 2 / (2 * d1 ** 2)) - np.exp(-y ** 2 / (2 * d2 ** 2))
                DoG[y + b][x + b] = ans
        return scipy.signal.convolve2d(image, DoG)


if __name__ == '__main__':
    image_2 = cv2.imread('images/image2.jpg', cv2.IMREAD_GRAYSCALE)
    image_2 = np.array(image_2, dtype='float32')
    plt.imshow(compute_LoG(image_2, 1), cmap='gray')
    plt.show()
    plt.imshow(compute_LoG(image_2, 2), cmap='gray')
    plt.show()
    for i in range(5):
        for j in range(5):
            plt.imshow(compute_LoG(image_2, 3, d1=i, d2=j), cmap='gray')
            plt.show()
