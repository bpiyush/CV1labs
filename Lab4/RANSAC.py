"""Performs RANSAC to find the best matches between an image pair."""

import numpy as np
import cv2

from keypoint_matching import KeypointMatcher
from utils import show_single_image, show_two_images


def project_2d_to_6d(X: np.ndarray):
    """Projects X (N x 2) to Z (2N x 6) space."""
    N = len(X)
    assert X.shape == (N, 2)

    Z = np.zeros((2 * N, 6))
    # in columns 0 to 2, fill even indexed rows of Z with X, and fill 5th column with 1
    Z[::2, 0:2] = X
    Z[::2, 4] = 1.0
    # in columns 2 to 4, fill odd indexed rows of Z with X
    Z[1::2, 2:4] = X
    Z[1::2, 5] = 1.0

    return Z


def project_6d_to_2d(Z: np.ndarray):
    """Projects Z (2N x 6) to X (N x 2) space."""
    N = len(Z) // 2
    assert Z.shape == (2 * N, 6)

    X_from_even_rows = Z[::2, 0:2]
    X_from_odd_rows = Z[1::2, 2:4]
    assert (X_from_even_rows == X_from_odd_rows).all()

    return X_from_even_rows



def project_2d_to_1d(X: np.ndarray):
    """Returns X (N x 2) from Z (2N, 1)"""
    N = len(X)
    X_stretched = np.zeros(2 * N)
    X_stretched[::2] = X[:, 0]
    X_stretched[1::2] = X[:, 1]
    return X_stretched


def project_1d_to_2d(Z: np.ndarray):
    """Returns X (N x 2) from Z (2N, 1)"""
    N = len(Z) // 2
    assert Z.shape == (2 * N,)

    X = np.zeros((N, 2))
    X[:, 0] = Z[::2]
    X[:, 1] = Z[1::2]

    return X


def rigid_body_transform(X: np.ndarray, params: np.ndarray):
    """Performs rigid body transformation of points X (N x 2) using params (6 x 1 flattened)"""
    N = len(X)
    assert X.shape == (N, 2)

    X = project_2d_to_6d(X)

    X_transformed = np.matmul(X, params)
    X_transformed = project_1d_to_2d(X_transformed)
    assert X_transformed.shape == (N, 2)

    return X_transformed


def rigid_body_transform_params(X1: np.ndarray, X2: np.ndarray):
    """Returns rigid-body transform parameters RT (6 x 1) assuming transformation between X1 and X2"""
    N = len(X1)
    assert X1.shape == X2.shape
    assert X1.shape == (N, 2)

    # X2 = X1 * params => params = psuedoinverse(X1) * X2
    X1_expanded = project_2d_to_6d(X1)
    assert X1_expanded.shape == (2 * N, 6)

    X2_stretched = project_2d_to_1d(X2)
    assert X2_stretched.shape == (2 * N,)

    params = np.dot(np.linalg.pinv(X1_expanded), X2_stretched)
    return params


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
        import ipdb; ipdb.set_trace()
        # check last row is [0, 0, 1]
        assert np.linalg.norm(np.linalg.norm(RT[-1, :] - np.array([1e-10, 1e-10, 1.0]))) < 1e-6

        # m1, m2, m3, m4 = RT[:-1, :-1].flatten()
        # t1, t2 = RT[:1, -1]

        # return m1, m2, m3, m4, t1, t2

        return RT, X1_homo, X2_homo

    def ransac(self, img1, kp1, img2, kp2, matches, num_matches=100, max_iter=10, radius_in_px=10, show_transformed=False):
        """Performs RANSAC to find best matches."""

        best_inlier_count = 0
        best_params = None

        for i in range(max_iter):
            # choose matches randomly
            selected_matches = np.random.choice(matches, num_matches)

            # get matched keypoints in img1
            X1 = np.array([kp1[matches[i].queryIdx].pt for i in range(len(matches))])

            # get matched keypoints in img2
            X2 = np.array([kp2[matches[i].trainIdx].pt for i in range(len(matches))])

            # get transformation parameters
            params = rigid_body_transform_params(X1, X2)
            
            # transform X1 to get X2_transformed
            X2_transformed = rigid_body_transform(X1, params)

            # find inliers
            diff = np.linalg.norm(X2_transformed - X2, axis=1)
            indices = diff < radius_in_px
            num_inliers = sum(indices)
            if num_inliers > best_inlier_count:
                print(f"Found {num_inliers} inliers!")
                best_params = params
                best_inlier_count = num_inliers

                # show matches
                selected_kp1 = [kp1[m.queryIdx] for m in selected_matches]
                selected_kp2 = [kp2[m.trainIdx] for m in selected_matches]
                selected_kp2_transformed = []
                _selected_matches = []
                for i in range(len(selected_matches)):
                    xy = cv2.KeyPoint(*X2_transformed[i].astype("uint8"), size=selected_kp2[i].size)
                    selected_kp2_transformed.append(xy)

                    _match = selected_matches[i]
                    _match.queryIdx = i
                    _match.trainIdx = i
                    _selected_matches.append(_match)

                img = cv2.drawMatches(img1, selected_kp1, img2, selected_kp2, _selected_matches, outImg=None, matchColor=0, matchesThickness=3, singlePointColor=(0, 0, 0))
                show_single_image(img)
                img = cv2.drawMatches(img1, selected_kp1, img2, selected_kp2_transformed, _selected_matches, outImg=img, matchColor=110, matchesThickness=3, singlePointColor=(0, 0, 0))
                show_single_image(img)
    
    def align(self, img1, kp1, img2, kp2, matches):
        self.ransac(img1, kp1, img2, kp2, matches)


def test_individual_functions():
    # checking rigid body transformation function
    X = np.array(
        [
            [0.0, 0.5],
            [0.5, 0.0],
            [0.5, 0.5],
            [0.1, 0.2],
            [0.8, 0.9],
            [-0.2, 1.0],
        ],
    )
    # 90-degree rotation
    RT = np.array(
        [
            [0.0, 1.0, 0.0],
            [-1., 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    Z = project_2d_to_6d(X)
    X_ = project_6d_to_2d(Z)

    params = np.hstack([RT[:2, :2].flatten(), RT[:2, 2].flatten()])
    X_transformed = rigid_body_transform(X, params)

    est_params = rigid_body_transform_params(X, X_transformed)
    assert np.linalg.norm(params - est_params) < 1e-8


if __name__ == "__main__":

    # testing individual functions
    test_individual_functions()

    # read & show images
    boat1 = cv2.imread('boat1.pgm', cv2.IMREAD_GRAYSCALE)
    boat2 = cv2.imread('boat2.pgm', cv2.IMREAD_GRAYSCALE)
    show_two_images(boat1, boat2, title="Sample pair of images.")

    # get matches
    kp_matcher = KeypointMatcher(contrastThreshold=0.1, edgeThreshold=5)
    matches, kp1, des1, kp2, des2 = kp_matcher.match(boat1, boat2, show_matches=True)

    image_alignment = ImageAlignment()
    image_alignment.align(boat1, kp1, boat2, kp2, matches)


