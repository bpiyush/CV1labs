import numpy as np
from gauss1D import gauss1D


def gauss2D( sigma_x, sigma_y , kernel_size ):
    ## solution
    G_x = gauss1D(sigma_x, kernel_size)
    G_y = gauss1D(sigma_x, kernel_size)
    G = np.outer(G_x, G_y)
    return G


if __name__ == "__main__":
    import cv2

    G = gauss2D(2, 2, 5)

    G_true = cv2.getGaussianKernel(ksize=5, sigma=2)
    G_true = np.outer(G_true, G_true)

    np.testing.assert_array_almost_equal(G, G_true, decimal=4)
