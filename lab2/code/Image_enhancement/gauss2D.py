import numpy as np
from gauss1D import gauss1D


def gauss2D(sigma_x, sigma_y, kernel_size):
    return np.outer(gauss1D(sigma_x, kernel_size), gauss1D(sigma_y, kernel_size))