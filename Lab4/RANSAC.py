"""Performs RANSAC to find the best matches between an image pair."""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from keypoint_matching import KeypointMatcher
from utils import show_single_image, show_two_images, show_three_images


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
    
    @staticmethod
    def show_transformed_points(img1, img2, X1, kp1, kp2, matches, params, num_inliers, num_to_show=20):

        H1, W1 = img1.shape
        H2, W2 = img2.shape
        img = np.hstack([img1, img2])

        random_matches = np.random.choice(matches, num_to_show)

        fig, ax = plt.subplots(1, 1, figsize=(15, 6))
        colors = cm.rainbow(np.linspace(0, 1, num_to_show))

        for i, match in enumerate(random_matches):

            # select a single match to visualize
            x1, y1 = kp1[match.queryIdx].pt
            x2, y2 = kp2[match.trainIdx].pt

            # get (x1, y1) transformed to (x1_transformed, y1_transformed)
            A = project_2d_to_6d(np.array([[x1, y1]]))
            (x1_transformed, y1_transformed) = np.dot(A, params)

            ax.imshow(img, cmap="gray")
            ax.axis("off")
            ax.scatter(x1_transformed + W1, y1_transformed, s=200, marker="x", color=colors[i])
            ax.plot(
                (x1, x1_transformed + W1), (y1, y1_transformed),
                linestyle="--", color=colors[i], marker="o",
            )

        ax.set_title(
            f"Points in image 1 mapped to transformed points estimated by {num_inliers} points.",
            fontsize=18,
        )

        plt.savefig(f"./results/match_transformed_inliers_{num_inliers}.png", bbox_inches="tight")
        plt.show()

    def ransac(
            self, img1, kp1, img2, kp2, matches, num_matches=6, max_iter=500,
            radius_in_px=10, show_transformed=True, inlier_th_for_show=1000
        ):
        """Performs RANSAC to find best matches."""

        best_inlier_count = 0
        best_params = None

        # get coordinates of all points in image 1
        X1 = np.array([kp1[matches[i].queryIdx].pt for i in range(len(matches))])

        # get coordinates of all points in image 2
        X2 = np.array([kp2[matches[i].trainIdx].pt for i in range(len(matches))])

        for i in range(max_iter):
            # choose matches randomly
            selected_matches = np.random.choice(matches, num_matches)

            # get matched keypoints in img1
            X1_selected = np.array([kp1[selected_matches[i].queryIdx].pt for i in range(len(selected_matches))])

            # get matched keypoints in img2
            X2_selected = np.array([kp2[selected_matches[i].trainIdx].pt for i in range(len(selected_matches))])

            # get transformation parameters
            params = rigid_body_transform_params(X1_selected, X2_selected)
            
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

                if show_transformed and num_inliers > inlier_th_for_show:
                    self.show_transformed_points(img1, img2, X1, kp1, kp2, matches, best_params, num_inliers)

        return best_params
    
    def align(
            self, img1, kp1, img2, kp2, matches, num_matches=6,
            max_iter=500, show_warped_image=True,
            save_warped=False, path="results/sample.png"
        ):
        best_params = self.ransac(img1, kp1, img2, kp2, matches, max_iter=max_iter, num_matches=num_matches)

        # apply the affine transformation using cv2.warpAffine()
        rows, cols = img1.shape[:2]

        M = np.zeros((2, 3))
        M[0, :2] = best_params[:2]
        M[1, :2] = best_params[2:4]
        M[0, 2] = best_params[4]
        M[1, 2] = best_params[5]

        img1_warped = cv2.warpAffine(img1, M, (cols, rows))

        if show_warped_image:
            show_three_images(
                img1, img2, img1_warped, title="",
                ax1_title="Image 1", ax2_title="Image 2", ax3_title="Transformation: Image 1 to Image 2",
            )

        if save_warped:
            plt.imsave(path, img1_warped)

        return best_params


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
    show_two_images(boat1, boat2, title="Given pair of images.")

    # get matches
    kp_matcher = KeypointMatcher(contrastThreshold=0.1, edgeThreshold=5)
    matches, kp1, des1, kp2, des2 = kp_matcher.match(boat1, boat2, show_matches=True)

    image_alignment = ImageAlignment()
    best_params = image_alignment.align(boat1, kp1, boat2, kp2, matches, show_warped_image=True)

    # experiment 1: varying number of maximum iterations
    run_expt = False
    np.random.seed(12345)
    iters = [10, 50, 100, 200, 500]
    for iter in iters:
        if run_expt:
            print(f"::::: Running alignment for max. {iter} iterations")
            best_params = image_alignment.align(
                boat1, kp1, boat2, kp2, matches, max_iter=iter,
                save_warped=True, path=f"results/img1_warped_iter_{iter}.png", show_warped_image=False,
            )

