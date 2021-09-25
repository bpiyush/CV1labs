import numpy as np


def gauss1D(sigma, kernel_size):
    G = np.zeros((1, kernel_size))
    if (kernel_size % 2 == 0):
        raise ValueError('kernel_size must be odd, otherwise the filter will not have a center to convolve on')
    # solution
    x = np.linspace(1, kernel_size, kernel_size)
    x = x - round(kernel_size / 2) - 1
    G = np.exp((-(x ** 2)) / ((2 * (sigma ** 2)))) / (np.sqrt(2 * np.pi) * sigma)
    G = G / np.sum(G)
    return G
