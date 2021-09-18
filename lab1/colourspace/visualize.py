import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


def imshow_helper(img, ax, title, cmap=None, xticks=False, yticks=False):
    ax.imshow(img, cmap=cmap)
    ax.set_title(title)
    if not xticks:
        ax.set_xticks([])
    if not yticks:
        ax.set_yticks([])


def visualize(input_image, img_type):
    # Fill in this function. Remember to remove the pass command

    img_type_to_channels = {
        "opponent": ["O1", "O2", "O3"],
        "rgb": ["r", "g", "b"],
        "ycbcr": ["Y", "Cb", "Cr"],
        "hsv": ["Hue", "Saturation", "Value"],
        "original": ["R", "G", "B"],
    }

    fig, ax = plt.subplots(1, 4, figsize=(10, 4))

    if img_type == "gray":
        # plot 4 variants of gray-scale image
        titles = ["Lightness", "Average", "Luminosity", "OpenCV"]
        assert input_image.shape[-1] == 4
        for i in range(input_image.shape[-1]):
            imshow_helper(input_image[..., i], ax[i], titles[i], cmap="gray")

    else:
        # set complete image
        imshow_helper(input_image, ax[0], "Converted image", cmap="gray")

        # set individual channels
        for i in range(3):
            imshow_helper(
                input_image[..., i], ax[i + 1],
                "Channel {}".format(img_type_to_channels[img_type][i]),
                cmap="gray"
            )

    fpath = "./{}.png".format(img_type)
    plt.savefig(fpath, bbox_inches="tight")

    plt.show()
