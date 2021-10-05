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