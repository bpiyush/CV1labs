import cv2
import numpy as np
import matplotlib.pyplot as plt


def denoise(image, kernel_type, k):
    if kernel_type == 'box':
        out = cv2.blur(src=image, ksize=(k, k))
    elif kernel_type == 'median':
        out = cv2.medianBlur(src=image, ksize=k)
    elif kernel_type == 'gaussian':
        out = cv2.GaussianBlur(src=image, ksize=(k, k), sigmaX=1.0)
    else:
        print('Operation not implemented')
    return out


if __name__ == '__main__':

    noise_names = {
        "saltpepper": "Salt And Pepper",
        "gaussian": "Gaussian",
    }

    kernel_type = "box"
    convert_to_float = False

    noise_type = "gaussian"
    noisy_image = cv2.imread(f'images/image1_{noise_type}.jpg', cv2.IMREAD_GRAYSCALE)

    ksizes = [3, 5, 7]
    denoised_outputs = []
    for k in ksizes:
        denoised_outputs.append(denoise(noisy_image, kernel_type, k))

    fig, ax = plt.subplots(1, 4, figsize=(20, 4), constrained_layout=True)

    ax[0].axis("off")
    ax[0].set_title(f"Noisy image ({noise_names[noise_type]})", fontsize=17)
    ax[0].imshow(noisy_image)

    for i, _ax in enumerate(ax[1:]):
        _ax.axis("off")
        _ax.imshow(denoised_outputs[i])
        _ax.set_title(f"${ksizes[i]}\\times{ksizes[i]}$", fontsize=17)
    
    plt.savefig(f"results/image1_{noise_type}_{kernel_type}.png", bbox_inches="tight")
    plt.show()
