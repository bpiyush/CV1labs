"""Common helper functions."""
from os import makedirs
from os.path import dirname
import numpy as np
import cv2
import matplotlib.pyplot as plt


def show_single_image(img, figsize=(7, 5), title="Single image"):
    """Displays a single image."""
    fig = plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(img)
    plt.title(title)
    plt.show()


def show_two_images(img1, img2, title="Two images"):
    """Displays a pair of images."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

    ax[0].axis("off")
    ax[0].imshow(img1)

    ax[1].axis("off")
    ax[1].imshow(img2)

    plt.suptitle(title)
    plt.show()


def show_three_images(img1, img2, img3, ax1_title="", ax2_title="", ax3_title="", title="Three images"):
    """Displays a triplet of images."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    ax[0].axis("off")
    ax[0].imshow(img1)
    ax[0].set_title(ax1_title)

    ax[1].axis("off")
    ax[1].imshow(img2)
    ax[1].set_title(ax2_title)

    ax[2].axis("off")
    ax[2].imshow(img3)
    ax[2].set_title(ax3_title)

    plt.suptitle(title)
    plt.show()


def show_many_images(
        images: list,
        grid: tuple = None,
        figsize=(18, 6),
        show=True,
        subtitles: list = None,
        suptitle: str = None,
        save: bool = False,
        save_path: str = None,
    ):
    """Shows even number of images"""
    # check even number of images
    assert len(images) % 2 == 0

    if grid is None:
        grid = (2, len(images) // 2)
    assert len(grid) == 2

    if subtitles is None:
        subtitles = [f"Image {i}" for i in range(len(images))]
    else:
        assert len(subtitles) == len(images)

    nrows, ncols = grid
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)

    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            _ax = ax[i, j]
            _ax.axis("off")
            _ax.imshow(images[idx])
            _ax.set_title(subtitles[idx])

    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=15, y = 1.06)

    if save is not None:
        assert save_path is not None
        assert isinstance(save_path, str)
        makedirs(dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()


def mark_kps_on_image(image, kps, color=(0, 255, 0), kps_with_size=True):

    if kps_with_size:
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    else:
        flags=0

    image_with_kps = cv2.drawKeypoints(
        image, kps, None, color=color,
        flags=flags,
    )
    return image_with_kps