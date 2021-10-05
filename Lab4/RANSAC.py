"""Performs RANSAC to find the best matches between an image pair."""

import numpy as np
import cv2

from keypoint_matching import KeypointMatcher
from utils import show_single_image, show_two_images


class ImageAlignment:
    """Class to perform alignment of a pair of images given keypoints."""

    def __init__(self) -> None:
        pass

    def get_transformation(self, kp1, kp2, matches):
        """Returns transformation parameters (rotation, translation) for given matches."""
        # get matched keypoints in img1
        X1 = np.array([kp1[matches[i].queryIdx].pt for i in range(len(matches))])
        # homogenize
        X1_homo = np.hstack([X1, np.ones((len(X1), 1))])

        # get matched keypoints in img2
        X2 = np.array([kp2[matches[i].trainIdx].pt for i in range(len(matches))])
        # homogenize
        X2_homo = np.hstack([X2, np.ones((len(X2), 1))])

        assert X1_homo.shape == X2_homo.shape

        # find transformation points
        RT = np.dot(np.linalg.pinv(X1_homo), X2_homo).T
        # check last row is [0, 0, 1]
        assert np.linalg.norm(np.linalg.norm(RT[-1, :] - np.array([1e-10, 1e-10, 1.0]))) < 1e-6

        # m1, m2, m3, m4 = RT[:-1, :-1].flatten()
        # t1, t2 = RT[:1, -1]

        # return m1, m2, m3, m4, t1, t2

        return RT, X1_homo, X2_homo

    def ransac(self, img1, kp1, img2, kp2, matches, num_matches=10, max_iter=100, radius_in_px=10, show_transformed=False):
        """Performs RANSAC to find best matches."""

        best_inlier_count = 0
        best_params = None

        for i in range(max_iter):
            # choose matches randomly
            selected_matches = np.random.choice(matches, num_matches)

            # get transformation parameters
            # m1, m2, m3, m4, t1, t2 = self.get_transformation(kp1, kp2, selected_matches)
            RT, X1_homo, X2_homo = self.get_transformation(kp1, kp2, selected_matches)

            # transform points
            X2_transformed_homo = np.dot(X1_homo, RT.T)

            # find inliers
            diff = np.linalg.norm(X2_transformed_homo - X2_homo, axis=1)
            indices = diff < radius_in_px
            num_inliers = sum(indices)
            if num_inliers > best_inlier_count:
                print(f"Found {num_inliers} inliers!")
                show_transformed = True
                best_params = RT

            if show_transformed:
                selected_kp1 = [kp1[m.queryIdx] for m in selected_matches]
                selected_kp2 = [kp2[m.trainIdx] for m in selected_matches]
                selected_kp2_transformed = []
                _selected_matches = []
                for i in range(len(selected_matches)):
                    xy = cv2.KeyPoint(*X2_transformed_homo[i, :2].astype("uint8"), size=selected_kp2[i].size)
                    selected_kp2_transformed.append(xy)

                    _match = selected_matches[i]
                    _match.queryIdx = i
                    _match.trainIdx = i
                    _selected_matches.append(_match)

                img = cv2.drawMatches(img1, selected_kp1, img2, selected_kp2, _selected_matches, outImg=None, matchColor=0, matchesThickness=3, singlePointColor=(0, 0, 0))
                show_single_image(img)
                img = cv2.drawMatches(img1, selected_kp1, img2, selected_kp2_transformed, _selected_matches, outImg=img, matchColor=110, matchesThickness=3, singlePointColor=(0, 0, 0))
                show_single_image(img)

        import ipdb; ipdb.set_trace()
        

    
    def align(self, img1, kp1, img2, kp2, matches):
        self.ransac(img1, kp1, img2, kp2, matches)


if __name__ == "__main__":
    # read & show images
    boat1 = cv2.imread('boat1.pgm', cv2.IMREAD_GRAYSCALE)
    boat2 = cv2.imread('boat2.pgm', cv2.IMREAD_GRAYSCALE)
    show_two_images(boat1, boat2, title="Sample pair of images.")

    # get matches
    kp_matcher = KeypointMatcher(contrastThreshold=0.1, edgeThreshold=5)
    matches, kp1, des1, kp2, des2 = kp_matcher.match(boat1, boat2, show_matches=True)

    image_alignment = ImageAlignment()
    image_alignment.align(boat1, kp1, boat2, kp2, matches)