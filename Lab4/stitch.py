"""Script to stitch a pair of images."""
import numpy as np
import cv2
import matplotlib.pyplot as plt

from keypoint_matching import KeypointMatcher
from RANSAC import ImageAlignment
from utils import show_two_images


if __name__ == "__main__":
    # read & show images
    left = cv2.imread('left.jpg')
    left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    left_gray = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    right = cv2.imread('right.jpg')
    right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
    right_gray = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
    show_two_images(left, right, title="Given pair of images.")

    # get matches
    kp_matcher = KeypointMatcher()
    matches, kp1, des1, kp2, des2 = kp_matcher.match(left_gray, right_gray, show_matches=True)
    print(len(matches))

    image_alignment = ImageAlignment()
    best_params = image_alignment.align(left_gray, kp1, right_gray, kp2, matches, show_warped_image=True, max_iter=1000, num_matches=100)

