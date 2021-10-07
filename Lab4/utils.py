"""Common helper functions."""
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