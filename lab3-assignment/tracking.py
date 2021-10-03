"""Script to implement feature tracking."""
from os.path import join, exists
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob

from lucas_kanade import lucas_kanade, lucas_kanade_for_points
from harris_corner_detector import harris_corner_detector, show_derivatives_and_corners


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

    print(f"::::::::::: Running tracking for {len(impaths)} images :::::::::::")
    I0 = read_image(impaths[0], convert=cv2.COLOR_BGR2GRAY)

    # detecting corners for first frame
    H, R, C = harris_corner_detector(I0)
    # show_derivatives_and_corners(I0, I0, I0, R, C, show=True)

    assert len(R) == len(C)
    points = np.array([[R[i], C[i]] for i in range(len(R))])

    # track in the first frame
    I1 = read_image(impaths[10], convert=cv2.COLOR_BGR2GRAY)
    V = lucas_kanade_for_points(I0, I1, points, make_plot=True)
    import ipdb; ipdb.set_trace()
