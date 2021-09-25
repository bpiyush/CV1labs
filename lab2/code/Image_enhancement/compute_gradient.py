import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import cv2
import matplotlib.pyplot as plt


def compute_gradient(image):
    Sx = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    Sy = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

    if image.dtype in [int, "uint8"]:
        image = image.astype("float32") / 255.0

    Gx = scipy.signal.convolve2d(image, Sx, mode='same')
    Gy = scipy.signal.convolve2d(image, Sy, mode='same')

    im_magnitude = np.sqrt(np.square(Gx) + np.square(Gy))
    assert np.isnan(im_magnitude).sum() == 0

    im_direction = np.arctan(Gy / (Gx + np.finfo(float).eps))

    return Gx, Gy, im_magnitude, im_direction


if __name__ == '__main__':

    image = cv2.imread("images/image2.jpg", cv2.IMREAD_GRAYSCALE)
    Gx, Gy, im_magnitude, im_direction = compute_gradient(image)
    
    images = [image, Gx, Gy, im_magnitude, im_direction]
    labels = ["Original image", "$G_x$", "$G_y$", "Magnitude", "Direction"]

    fig, ax = plt.subplots(1, 5, figsize=(20, 5), constrained_layout=True)
    for i, _ax in enumerate(ax):
        _ax.axis("off")
        _ax.imshow(images[i])
        _ax.set_title(labels[i], fontsize=18)

    plt.savefig("results/gradient.png", bbox_inches="tight")
    plt.show()
