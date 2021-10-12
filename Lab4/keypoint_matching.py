"""Implements keypoint matching for a pair of images."""

import numpy as np
import cv2

from utils import show_single_image, show_two_images


class KeypointMatcher:
    """Class for Keypoint matching for a pair of images."""

    def __init__(self, **sift_args) -> None:
        self.SIFT = cv2.SIFT_create(**sift_args)
        self.BFMatcher = cv2.BFMatcher()
    
    @staticmethod
    def _check_images(img1: np.ndarray, img2: np.ndarray):
        assert isinstance(img1, np.ndarray)
        assert len(img1.shape) == 2

        assert isinstance(img2, np.ndarray)
        assert len(img2.shape) == 2

        # assert img1.shape == img2.shape
    
    @staticmethod
    def _show_matches(img1, kp1, img2, kp2, matches, K=10, figsize=(10, 5), drawMatches_args=dict(matchesThickness=3, singlePointColor=(0, 0, 0))):
        """Displays matches found in the image"""
        selected_matches = np.random.choice(matches, K)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, selected_matches, outImg=None, **drawMatches_args)
        show_single_image(img3, figsize=figsize, title=f"Randomly selected K = {K} matches between the pair of images.")
        return img3

    def match(self, img1: np.ndarray, img2: np.ndarray, show_matches: bool = True):
        """Finds, describes and matches keypoints in given pair of images."""
        # check input images
        self._check_images(img1, img2)

        # find kps and descriptors in each image
        kp1, des1 = self.SIFT.detectAndCompute(img1, None)
        kp2, des2 = self.SIFT.detectAndCompute(img2, None)

        # compute matches via Brute-force matching
        matches = self.BFMatcher.match(des1, des2)

        # sort them in the order of their distance
        matches = sorted(matches, key = lambda x:x.distance)

        if show_matches:
            self._show_matches(img1, kp1, img2, kp2, matches)

        return matches, kp1, des1, kp2, des2


if __name__ == "__main__":
    # read & show images
    boat1 = cv2.imread('boat1.pgm', cv2.IMREAD_GRAYSCALE)
    boat2 = cv2.imread('boat2.pgm', cv2.IMREAD_GRAYSCALE)
    show_two_images(boat1, boat2, title="Given pair of images.")

    kp_matcher = KeypointMatcher(contrastThreshold=0.1, edgeThreshold=5)
    matches, kp1, des1, kp2, des2 = kp_matcher.match(boat1, boat2, show_matches=True)