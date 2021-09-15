"""Script to implement Gray World Algorithm"""
from PIL.Image import new
import numpy as np
import cv2
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
from photometric.utils import show_single_image, show_multiple_images


def gray_world(img: np.ndarray) -> np.ndarray:
    """Applies gray-world algo to an image to filter it of color artefacts.

    Args:
        img (np.ndarray): image read with channels RGB
    """
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # compute channel-wise means and global mean
    B_mean = np.mean(B)
    G_mean = np.mean(G)
    R_mean = np.mean(R)
    net_mean = np.mean([B_mean, G_mean, R_mean])

    # normalize R channel
    R_new = (net_mean / (R_mean + np.finfo(float).eps)) * R
    R_new = np.minimum(R_new.astype(int), 255)

    # normalize B channel
    B_new = (net_mean / (B_mean + np.finfo(float).eps)) * B
    B_new = np.minimum(B_new.astype(int), 255)

    # normalize G channel
    G_new = (net_mean / (G_mean + np.finfo(float).eps)) * G
    G_new = np.minimum(G_new.astype(int), 255)

    new_img = np.dstack([R_new, G_new, B_new])
    new_img = new_img.astype("uint8")
    
    return new_img


if __name__ == "__main__":
    img = np.zeros((512, 512, 3))
    new_img = gray_world(img)
    assert new_img.shape == (512, 512, 3)

    # load input image
    img = cv2.imread("awb.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()

    # run the algo for this image
    new_img = gray_world(img)
    plt.imshow(new_img)
    plt.show()

