"""Script to implement Gray World Algorithm"""
from PIL.Image import new
import numpy as np
import cv2
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
from photometric.utils import show_single_image, show_multiple_images


def gray_world(img: np.ndarray):
    """Applies gray-world algo to an image to filter it of color artefacts.

    Args:
        img (np.ndarray): image read with channels RGB
    """
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    B_mean = np.mean(B)
    G_mean = np.mean(G)
    R_mean = np.mean(R)

    net_mean = np.mean([B_mean, G_mean, R_mean])

    R_new = (net_mean / (R_mean + np.finfo(float).eps)) * R
    R_new = np.minimum(R_new.astype(int), 255)
    B_new = (net_mean / (B_mean + np.finfo(float).eps)) * B
    B_new = np.minimum(B_new.astype(int), 255)
    G_new = (net_mean / (G_mean + np.finfo(float).eps)) * G
    G_new = np.minimum(G_new.astype(int), 255)

    new_img = np.dstack([R_new, G_new, B_new])
    new_img = new_img.astype("uint8")
    
    return new_img


def grey_world_aliter(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    mu_g = np.average(nimg[1])
    nimg[0] = np.minimum(nimg[0]*(mu_g/np.average(nimg[0])),255)
    nimg[2] = np.minimum(nimg[2]*(mu_g/np.average(nimg[2])),255)
    return  nimg.transpose(1, 2, 0).astype(np.uint8)


if __name__ == "__main__":
    # img = np.zeros((512, 512, 3))
    # new_img = gray_world(img)

    # test on given sample image
    img = cv2.imread("awb.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    show_single_image(img)

    new_img = gray_world(img)
    show_single_image(new_img)

    new_img = grey_world_aliter(img)
    show_single_image(new_img)

