import numpy as np

def gauss1D(sigma , kernel_size):
    G = np.zeros((1, kernel_size))
    if kernel_size % 2 == 0:
        raise ValueError('kernel_size must be odd, otherwise the filter will not have a center to convolve on')
    # solution

    xlim = np.floor(kernel_size / 2.0)
    x = np.arange(-xlim, xlim + 1, 1)
    G = np.exp((-(x**2))/((2*(sigma**2))))/ (np.sqrt(2*np.pi)*sigma)
    G = G /np.sum(G)

    return G


if __name__ == "__main__":
    G = gauss1D(2, 5)
    G_true = np.array([0.1525, 0.2218, 0.2514, 0.2218, 0.1525])
    np.testing.assert_array_almost_equal(G, G_true, decimal=4)
