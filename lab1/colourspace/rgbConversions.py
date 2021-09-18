import numpy as np
import cv2

def rgb2grays(input_image):
    # converts an RGB into grayscale by using 4 different methods

    # convert to [0. 1] to make sure everything remains within [0, 1]
    input_image /= 255.0
    R, G, B = np.rollaxis(input_image, 2)

    # ligtness method
    lightness_new_image = (np.max(input_image, axis=2) + np.min(input_image, axis=2)) / 2.0
    lightness_new_image = (lightness_new_image * 255).astype("uint8")

    # average method
    average_new_image = np.sum(input_image, axis=2) / 3
    average_new_image = (average_new_image * 255).astype("uint8")

    # luminosity method
    luminosity_new_image = 0.21 * R + 0.72 * G + 0.07 * B
    luminosity_new_image = (luminosity_new_image * 255).astype("uint8")

    # built-in opencv function 
    cv2_new_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    # cv2_new_image = (cv2_new_image * 255).astype("uint8")

    # send all 4 back as 4-channels
    new_image = np.dstack(
        [lightness_new_image, average_new_image, luminosity_new_image, cv2_new_image]
    )

    return new_image


def rgb2opponent(input_image):
    # converts an RGB image into opponent colour space
    input_image /= 255.0

    matrix = np.array(
        [
            [1.0 / np.sqrt(2), -1.0 / np.sqrt(2), 0],
            [1.0 / np.sqrt(6), 1.0 / np.sqrt(6), -2.0 / np.sqrt(6)],
            [1.0 / np.sqrt(3), 1.0 / np.sqrt(3), 1.0 / np.sqrt(3)]
        ]
    )
    new_image = np.dot(input_image, matrix)

    return new_image


def rgb2normedrgb(input_image):
    # converts an RGB image into normalized rgb colour space
    input_image /= 255.0
    R, G, B = np.rollaxis(input_image, axis=2)
    norm_sum = R + G + B

    r = R / norm_sum
    g = G / norm_sum
    b = B / norm_sum

    new_image = np.dstack([r, g, b])

    return new_image
