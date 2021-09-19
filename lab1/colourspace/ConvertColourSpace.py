from matplotlib.pyplot import colormaps
import numpy as np
import cv2
import rgbConversions
from visualize import *

def ConvertColourSpace(input_image, colourspace):
    '''
    Converts an RGB image into a specified color space, visualizes the
    color channels and returns the image in its new color space.

    Colorspace options:
      opponent
      rgb -> for normalized RGB
      hsv
      ycbcr
      gray

    P.S: Do not forget the visualization part!
    '''

    # Convert the image into double precision for conversions
    input_image = input_image.astype(np.float32)

    if colourspace.lower() == 'opponent':
        # fill in the rgb2opponent function
        new_image = rgbConversions.rgb2opponent(input_image)

    elif colourspace.lower() == 'rgb':
        # fill in the rgb2opponent function
        new_image = rgbConversions.rgb2normedrgb(input_image)

    elif colourspace.lower() == 'hsv':
        # use built-in function from opencv
        # opencv expects the input RGB image to be in [0, 1]
        new_image = cv2.cvtColor(input_image / 255.0, cv2.COLOR_RGB2HSV)
        # normalize the H channel since it has values in [0, 360]
        new_image[..., 0] /= new_image[..., 0].max()

    elif colourspace.lower() == 'ycbcr':
        # use built-in function from opencv
        new_image = cv2.cvtColor(input_image / 255.0, cv2.COLOR_RGB2YCrCb)
        # convert YCrCb -> YCbCr
        new_image[..., [1, 2]] = new_image[..., [2, 1]]

    elif colourspace.lower() == 'gray':
        # fill in the rgb2opponent function
        new_image = rgbConversions.rgb2grays(input_image)

    else:
        print('Error: Unknown colorspace type [%s]...' % colourspace)
        new_image = input_image.astype("uint8")

    visualize(new_image, colourspace.lower())

    return new_image


if __name__ == '__main__':
    # Replace the image name with a valid image
    img_path = '../colour_constancy/awb.jpg'
    # Read with opencv
    I = cv2.imread(img_path)
    # Convert from BGR to RGB
    # This is a shorthand.
    I = I[:, :, ::-1]

    # # testing: original image with no conversion
    # out_img = ConvertColourSpace(I, 'original')

    # type 1: conversion to `opponent`
    # out_img = ConvertColourSpace(I, 'opponent')

    # # type 2: conversion to rgb
    # out_img = ConvertColourSpace(I, 'rgb')

    # # type 3: conversion to ycbcr
    # out_img = ConvertColourSpace(I, 'YCbCr')

    # # type 4: conversion to hsv
    out_img = ConvertColourSpace(I, 'hsv')

    # # type 5: conversion to gray
    # out_img = ConvertColourSpace(I, 'gray')
