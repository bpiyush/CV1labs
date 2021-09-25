import numpy as np
import cv2


def myPSNR(orig_image, approx_image):
    imax = np.max(orig_image)
    mse = np.square(orig_image - approx_image)
    mse = np.sum(mse)
    mse /= np.prod(orig_image.shape)
    PSNR = 20*np.log10(imax / (np.sqrt(mse) + np.finfo(float).eps))
    return PSNR


if __name__ == '__main__':
    salt_and_pepper = cv2.imread('images/image1_saltpepper.jpg', cv2.IMREAD_GRAYSCALE)
    image_one = cv2.imread('images/image1.jpg', cv2.IMREAD_GRAYSCALE)
    gaussian = cv2.imread('images/image1_gaussian.jpg', cv2.IMREAD_GRAYSCALE)
    salt_and_pepper = np.array(salt_and_pepper, dtype='float32')
    image_one = np.array(image_one, dtype='float32')
    gaussian = np.array(gaussian, dtype='float32')

    print(myPSNR(salt_and_pepper, image_one))
    print(myPSNR(gaussian, image_one))

