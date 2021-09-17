"""
Script to perform reconstruction of an image using its intrinsic decompositions,
here, we consider albedo and shading.
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt


def form_image_from_decompositions(albedo: np.ndarray, shading: np.ndarray) -> np.ndarray:
    """Forms image given its albedo and shading via ele-wise multiplication.

    Args:
        albedo (np.ndarray): albedo or reflectance
        shading (np.ndarray): shading or illuminance

    Returns:
        np.ndarray: reconstructed image (in range 0 to 255 as type int)
    """
    assert albedo.shape == shading.shape
    assert albedo.dtype == shading.dtype

    if albedo.dtype in ["uint8", "int"]:
        # need to divide by 255 first (else, after multiplication values will go >> 255)
        assert albedo.max() <= 255 and albedo.min() >= 0
        assert shading.max() <= 255 and shading.min() >= 0

        albedo = albedo.astype("float") / 255.0
        shading = shading.astype("float") / 255.0

        return (np.multiply(albedo, shading) * 255).astype("uint8")
    elif albedo.dtype == "float":
        # can safely multiply since (< 1) * (< 1) -> < 1
        assert albedo.max() <= 1.0 and albedo.min() >= 0
        assert shading.max() <= 1.0 and shading.min() >= 0

        return (np.multiply(albedo, shading) * 255).astype("uint8")
    else:
        raise ValueError("albedo.dtype and shading.dtype must be either int or float")


if __name__ == "__main__":

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    oimg = cv2.imread("./ball.png")
    oimg = cv2.cvtColor(oimg, cv2.COLOR_BGR2RGB)

    albedo = cv2.imread("./ball_albedo.png")
    albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)

    shading = cv2.imread("./ball_shading.png")
    shading = cv2.cvtColor(shading, cv2.COLOR_BGR2RGB)

    nimg = form_image_from_decompositions(albedo, shading)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(oimg)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_xlabel("Original image", fontsize=20)

    ax[1].imshow(albedo)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_xlabel("Albedo", fontsize=20)

    ax[2].imshow(shading)
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].set_xlabel("Shading", fontsize=20)

    ax[3].imshow(nimg)
    ax[3].set_xticks([])
    ax[3].set_yticks([])
    ax[3].set_xlabel("Reconstructed image", fontsize=20)

    plt.savefig("./results/iid.png", bbox_inches="tight")
    plt.show()