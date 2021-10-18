"""Common helper functions."""
from os import makedirs
from os.path import dirname
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt

from constants import idx_to_class


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
            _ax.set_title(subtitles[idx], fontsize=15)

    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=20, y = 1.06)

    if save:
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


def load_pkl(path: str, encoding: str = "ascii"):
    """Loads a .pkl file.
    Args:
        path (str): path to the .pkl file
        encoding (str, optional): encoding to use for loading. Defaults to "ascii".
    Returns:
        Any: unpickled object
    """
    return pickle.load(open(path, "rb"), encoding=encoding)


def save_pkl(data, path: str) -> None:
    """Saves given object into .pkl file
    Args:
        data (Any): object to be saved
        path (str): path to the location to be saved at
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def print_update(message: str, width: int = 100, fillchar: str = ":") -> str:
    """Prints an update message
    Args:
        message (str): message
        width (int): width of new update message
        fillchar (str): character to be filled to L and R of message
    Returns:
        str: print-ready update message
    """
    message = message.center(len(message) + 2, " ")
    print("\n" + message.center(width, fillchar) + "\n")


def plot_feature_histograms(features, labels, K=500, save=False, save_path="./results/feature_hist.png", show=True):
    """Plots feature histogram for each class."""
    assert len(features) == len(labels)
    N, K = features.shape
    x = np.arange(0, K, 1)

    classes = np.sort(np.unique(labels))
    n_classes = len(classes)

    fig, ax = plt.subplots(1, n_classes, figsize=(24, 5), constrained_layout=True)
    
    for i, c in enumerate(classes):
        class_indices = np.where(labels == c)
        class_features = features[class_indices]
        class_features = np.mean(class_features, axis=0)

        ax[i].grid()
        ax[i].bar(x=x, height=class_features, ec="cornflowerblue", fc="cornflowerblue", alpha=1.0, linewidth=1.0)
        ax[i].set_title(idx_to_class[c].capitalize())

        if i == (n_classes // 2):
            ax[i].set_xlabel("Index of visual vocabulary (of size K)", fontsize=18)
        
        if i == 0:
            ax[i].set_ylabel("Normalized frequency counts", fontsize=18)

    plt.suptitle(f"Normalized Feature histograms per class for K={K}", fontsize=22)
    if save:
        assert save_path is not None
        makedirs(dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
