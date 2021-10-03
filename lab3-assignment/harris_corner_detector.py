"""Script to detect Harris corners in an image."""
from os.path import basename
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter

from utils import show_multiple_images


def _check_image(I):
    assert isinstance(I, np.ndarray)
    assert len(I.shape) == 2
    H, W = I.shape
    return H, W


def compute_Ix(I: np.ndarray, sigma: float = 1.0):
    gauss_1d = cv2.getGaussianKernel(ksize=3, sigma=sigma)
    Gx = np.array([-1.0, 0.0, 1.0])
    smoooth_Gx = np.multiply(Gx, gauss_1d)
    Ix = cv2.filter2D(src=I, ddepth=-1, kernel=smoooth_Gx)
    return Ix


def compute_Iy(I: np.ndarray, sigma: float = 1.0):
    gauss_1d = cv2.getGaussianKernel(ksize=3, sigma=sigma)
    Gy = np.array([-1.0, 0.0, 1.0])
    smoooth_Gy = np.multiply(Gy, gauss_1d).T
    Iy = cv2.filter2D(src=I, ddepth=-1, kernel=smoooth_Gy)
    return Iy


def get_corners(H: np.ndarray, threshold: float, window_size: int):
    H_local_max = maximum_filter(H, size=window_size)
    H[H < H_local_max] = 0.0
    r, c = np.where(H > threshold)
    return r, c


def harris_corner_detector(
        I: np.ndarray,
        gauss_sigma: float = 1.0,
        threshold: float = 0.001,
        debug: bool = False,
        window_size: int = 5,
    ):
    h, w = _check_image(I)

    # computed Gaussian-smoothed derivative along x-axis
    Ix = compute_Ix(I, sigma=gauss_sigma)

    # computed Gaussian-smoothed derivative along y-axis
    Iy = compute_Iy(I, sigma=gauss_sigma)

    # compute second order derivative terms
    IxIx = Ix ** 2
    IyIy = Iy ** 2
    IxIy = np.multiply(Ix, Iy)

    # compute elements of Q matrix
    gauss_1d = cv2.getGaussianKernel(ksize=3, sigma=gauss_sigma)
    gauss_2d = np.outer(gauss_1d, gauss_1d)
    A = cv2.filter2D(src=IxIx, ddepth=-1, kernel=gauss_2d)
    B = cv2.filter2D(src=IxIy, ddepth=-1, kernel=gauss_2d)
    C = cv2.filter2D(src=IyIy, ddepth=-1, kernel=gauss_2d)

    # compute H
    eigen_mul = np.multiply(A, C) - B ** 2
    eigen_sum = A + C
    H = eigen_mul - 0.04 * (eigen_sum ** 2)

    if debug:
        show_multiple_images([I, Ix, Iy, IxIx, IyIy, IxIy, H], grid=(1, 7), figsize=(20, 4), show=True)

    r, c = get_corners(H, threshold=threshold, window_size=window_size)
    
    return H, r, c


def show_derivatives_and_corners(I, Ix, Iy, r, c, save=False, path="results/sample.png", show=False):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    ax[-1].axis("off")
    ax[-1].imshow(I)
    ax[-1].scatter(c, r, color="red", s=10, marker="o")
    ax[-1].set_title("$I$ with Harris corners", fontsize=18)

    ax[0].axis("off")
    ax[0].imshow(Ix)
    ax[0].set_title("$I_x$", fontsize=18)

    ax[1].axis("off")
    ax[1].imshow(Iy)
    ax[1].set_title("$I_y$", fontsize=18)

    if save:
        # path = f"./results/harris_{basename(impath).split('.')[0]}.png"
        plt.savefig(path, bbox_inches="tight")

    if show:
        plt.show()


def rotate_image(I, angle):
    """
    Rotates image by a given angle (in degrees).

    Inspired from: https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
    """
    h, w = _check_image(I)
    I_center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(I_center, angle, 1.0)
    result = cv2.warpAffine(I, rot_mat, (w, h), flags=cv2.INTER_LINEAR)
    return result


def demo(impath: str, show: bool = False):
    """Runs Harris feature detection on a given image and displays result."""
    I = cv2.imread(impath)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    I = I.astype(float) / 255.0

    Ix = compute_Ix(I)
    Iy = compute_Iy(I)
    H, r, c = harris_corner_detector(I, debug=False)

    show_derivatives_and_corners(I, Ix, Iy, r, c, show=show)

    return H, r, c, I, Ix, Iy

if __name__ == "__main__":
    impath = "./images/toy/0001.jpg"
    # impath = "./images/doll/0200.jpg"

    H, r, c, I, Ix, Iy = demo(impath, show=True)

    # experiment 1: varying threshold
    thresholds = [0.0001, 0.001, 0.002, 0.005, 0.01]
    R, C = [], []
    for threshold in thresholds:
        H, r, c = harris_corner_detector(I, threshold=threshold)
        R.append(r)
        C.append(c)
    
    fig, ax = plt.subplots(1, len(thresholds), figsize=(5 * len(thresholds), 5))
    for i, th in enumerate(thresholds):
        _ax = ax[i]
        _ax.axis("off")
        _ax.set_title(f"Threshold: {th}", fontsize=15)
        _ax.imshow(I)
        _ax.scatter(C[i], R[i], color="red", s=10, marker="o")
    
    save_path = f"./results/harris_threshold_{basename(impath).split('.')[0]}.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    # experiment 2: checking rotation invariance
    H, r, c = harris_corner_detector(I, debug=False)

    # rotate by 45 degrees
    I_45 = rotate_image(I, angle=45)
    H_45, r_45, c_45 = harris_corner_detector(I_45, debug=False)

    # rotate by 90 degrees
    I_90 = rotate_image(I, angle=90)
    H_90, r_90, c_90 = harris_corner_detector(I_90, debug=False)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    ax[0].axis("off")
    ax[0].imshow(I)
    ax[0].scatter(c, r, color="red", s=10, marker="o")
    ax[0].set_title("Original $I$", fontsize=15)

    ax[1].axis("off")
    ax[1].imshow(I_45)
    ax[1].scatter(c_45, r_45, color="red", s=10, marker="o")
    ax[1].set_title("$I$ rotated by $45^{o}$", fontsize=15)

    ax[2].axis("off")
    ax[2].imshow(I_90)
    ax[2].scatter(c_90, r_90, color="red", s=10, marker="o")
    ax[2].set_title("$I$ rotated by $90^{o}$", fontsize=15)

    save_path = f"./results/harris_rotation_{basename(impath).split('.')[0]}.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

