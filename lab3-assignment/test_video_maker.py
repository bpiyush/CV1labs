"""Test script for generating a video"""
from os.path import join, exists
from glob import glob
import cv2
import numpy as np


def read_image(impath, convert=cv2.COLOR_BGR2RGB, normalize=True):
    assert exists(impath), f"Given image does not exist at {impath}"

    I = cv2.imread(impath)

    if convert is not None:
        I = cv2.cvtColor(I, convert)

    if normalize:
        I = I.astype(float) / 255.0
    
    return I


if __name__ == "__main__":
    example = "toy"
    impaths = glob(join("images", example, "*.jpg"))
    impaths = sorted(impaths)
    images = [read_image(x, convert=cv2.COLOR_BGR2GRAY, normalize=False) for x in impaths]

    height, width = images[0].shape[0], images[0].shape[1]
    writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 5,(width, height))
    for frame in images:
        frame = np.expand_dims(frame, 2)
        frame = np.repeat(frame, 3, axis=2)
        writer.write(frame.astype("uint8"))
    writer.release()