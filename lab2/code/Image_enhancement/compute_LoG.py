import cv2
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import math


def mask(n):
    mask = np.ones((n, n))
    mask[n // 2][n // 2] = 1 - n ** 2
    return mask


def LoG_at_pixel(x, y, sigma):
    """Computes the Laplacian of Gaussian at a given (x, y)"""
    value = (x ** 2 + y ** 2) / (2.0 * (sigma ** 2))
    return (-1.0 / (np.pi * sigma ** 4)) * np.exp(-value) * (1 - value)


def LoG_kernel(sigma, ksize):
    """Computes the Laplacian of Gaussian at a given (x, y)"""
    log = np.zeros((ksize, ksize))
    b = int(ksize / 2)
    for x in np.arange(-b, b + 1, 1):
        for y in np.arange(-b, b + 1, 1):
            log[y +b][x + b] = LoG_at_pixel(x=x, y=y, sigma=sigma)
    return log


def Gaussian_at_pixel(x, y, sigma):
    """Computes the Gaussian at a given (x, y)"""
    value = (x ** 2 + y ** 2) / (2.0 * (sigma ** 2))
    return (1.0 / (2 * np.pi * sigma ** 2)) * np.exp(-value)


def Gaussian_kernel(sigma, ksize):
    """Computes the Gaussian at a given (x, y)"""
    gauss = np.zeros((ksize, ksize))
    b = int(ksize / 2)
    for x in np.arange(-b, b + 1, 1):
        for y in np.arange(-b, b + 1, 1):
            gauss[y +b][x + b] = Gaussian_at_pixel(x=x, y=y, sigma=sigma)
    return gauss


def compute_LoG(image, LOG_type, d1=1.0, d2=1.0):
    if LOG_type == 1:
        gauss = Gaussian_kernel(sigma=0.5, ksize=5)
        smoothed = scipy.signal.convolve2d(image, gauss/np.sum(gauss))
        laplacian = LoG_kernel(sigma=0.5, ksize=5)
        return scipy.signal.convolve2d(smoothed, laplacian/np.sum(laplacian))

    elif LOG_type == 2:
        laplacian = LoG_kernel(sigma=0.5, ksize=5)
        return scipy.signal.convolve2d(image, laplacian/np.sum(laplacian))

    elif LOG_type == 3:
        gauss_1 = Gaussian_kernel(sigma=0.4, ksize=5)
        gauss_2 = Gaussian_kernel(sigma=1.0, ksize=5)
        dog = gauss_1 - gauss_2

        return scipy.signal.convolve2d(image, dog/np.sum(dog))


if __name__ == '__main__':
    image_2 = cv2.imread('images/image2.jpg', cv2.IMREAD_GRAYSCALE)
    image_2 = np.array(image_2 / 255.0, dtype='float32')

    labels = ["Laplacian", "LoG", "DoG"]

    fig, ax = plt.subplots(1, 4, figsize=(15, 3), constrained_layout=True)
    ax[0].axis("off")
    ax[0].imshow(image_2)
    ax[0].set_title("Original image")

    for i, _ax in enumerate(ax[1:]):
        _ax.imshow(compute_LoG(image_2, i + 1))
        _ax.axis("off")
        _ax.set_title(labels[i])
    
    plt.savefig("results/log_all.png", bbox_inches="tight")
    plt.show()
