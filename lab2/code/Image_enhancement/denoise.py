import cv2
import numpy as np
import matplotlib.pyplot as plt


def denoise(image, kernel_type, k, sigma):
    if kernel_type == 'box':
        out = cv2.blur(src=image, ksize=(k, k))
    elif kernel_type == 'median':
        out = cv2.medianBlur(src=image, ksize=k)
    elif kernel_type == 'gaussian':
        out = cv2.GaussianBlur(src=image, ksize=(k, k), sigmaX=sigma)
    else:
        print('Operation not implemented')
    return out


if __name__ == '__main__':
    from myPSNR import myPSNR

    original_image = cv2.imread(f'images/image1.jpg', cv2.IMREAD_GRAYSCALE)

    noise_names = {
        "saltpepper": "Salt And Pepper",
        "gaussian": "Gaussian",
    }

    kernel_type = "gaussian"

    noise_type = "gaussian"
    noisy_image = cv2.imread(f'images/image1_{noise_type}.jpg', cv2.IMREAD_GRAYSCALE)

    ksizes = [3, 5, 7]
    denoised_outputs = []
    for k in ksizes:
        denoised_image = denoise(noisy_image, kernel_type, k)
        denoised_outputs.append(denoised_image)

        psnr = myPSNR(original_image.astype("float32"), denoised_image.astype("float32"))
        print(f"PSNR for ({kernel_type}, {noise_type}, filter size: {k}): {psnr}")

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

    # experiment for Gaussian filter for Gaussian noised images
    original_image = cv2.imread(f'images/image1.jpg', cv2.IMREAD_GRAYSCALE)

    kernel_type = "gaussian"

    noise_type = "gaussian"
    noisy_image = cv2.imread(f'images/image1_{noise_type}.jpg', cv2.IMREAD_GRAYSCALE)

    sigmas = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    ksize = 5
    denoised_outputs = []
    for sigma in sigmas:
        denoised_image = denoise(noisy_image, kernel_type, k=ksize, sigma=sigma)
        denoised_outputs.append(denoised_image)

        psnr = myPSNR(original_image.astype("float32"), denoised_image.astype("float32"))
        print(f"PSNR for ({kernel_type}, {noise_type}, filter size: {ksize}, sigma: {sigma}): {psnr}")

    fig, ax = plt.subplots(1, len(sigmas) + 1, figsize=(20, 4), constrained_layout=True)

    ax[0].axis("off")
    ax[0].set_title(f"Noisy image", fontsize=17)
    ax[0].imshow(noisy_image)

    for i, _ax in enumerate(ax[1:]):
        _ax.axis("off")
        _ax.imshow(denoised_outputs[i])
        _ax.set_title(f"$\sigma = {sigmas[i]}$", fontsize=17)
    
    plt.savefig(f"results/image1_{noise_type}_{kernel_type}_sigmas.png", bbox_inches="tight")
    plt.show()
