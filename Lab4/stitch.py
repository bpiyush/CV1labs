"""Script to stitch a pair of images."""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

from keypoint_matching import KeypointMatcher
from RANSAC import ImageAlignment, project_2d_to_6d, project_1d_to_2d
from utils import show_two_images




class ImageStitching:
    """Class to perform stitching of a pair of images."""

    def __init__(self) -> None:
        np.random.seed(1)
        pass

    def _get_alignment_params(self, img1, img2, show_matches=True, kpmatch_args=dict(), align_args=dict()):
        """Detects keypoints, performs matching and returns matches."""
        kp_matcher = KeypointMatcher(**kpmatch_args)
        matches, kp1, des1, kp2, des2 = kp_matcher.match(img1, img2, show_matches=show_matches)

        image_alignment = ImageAlignment(**align_args)
        best_params = image_alignment.align(
            img1, kp1, img2, kp2, matches,
            show_warped_image=True, num_matches=4, max_iter=2000, method="cv2",
        )

        return matches, kp1, des1, kp2, des2, best_params
    
    def _get_transform_matrix_and_inv_matrix(self, params: np.ndarray):
        assert params.shape == (6,)
        AM = np.zeros((3, 3))
        AM[0, :2] = params[:2]
        AM[1, :2] = params[2:4]
        AM[0, 2] = params[4]
        AM[1, 2] = params[5]
        AM[-1, -1] = 1.0

        M_1_to_2 = AM[:2]
        M_2_to_1 = np.linalg.inv(AM)[:2]

        return M_1_to_2, M_2_to_1
    
    @staticmethod
    def show_canvas(canvas, subcanvas, C2_as_viewed_from_1, X_left, Y_top, W_new, H_new):
        """Displays canvas with both images and transformed image 2 in image 1 space."""
        fig, ax = plt.subplots(1, 1, figsize=(13, 7), dpi=100)

        ax.imshow(canvas.astype(int), alpha=0.6)
        ax.imshow(subcanvas.astype(int), alpha=0.4)
        ax.scatter(C2_as_viewed_from_1[:, 0], C2_as_viewed_from_1[:, 1], c="red", label="Corners of Image 2 transformed")

        stiched_image_rect = Rectangle(
            (X_left, Y_top), W_new, H_new, fc='none', ec='limegreen', lw=1, label="Border of stiched image",
        )
        ax.add_patch(stiched_image_rect)

        title = "Image 1 (Left), Image 2 (Right) and Image 2 transformed (middle) based on estimated affine parameters."
        ax.set_title(title, fontsize=13)

        plt.legend(loc="lower right")
        plt.show()
    
    def stitch(self, img1, img2):
        """Main stitching function. NOTE: this expects RGB images."""
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

        # get matches and alignment parameters
        matches, kp1, des1, kp2, des2, best_params = self._get_alignment_params(img1_gray, img2_gray)

        # get transformation params as matrices
        M_1_to_2, M_2_to_1 = self._get_transform_matrix_and_inv_matrix(best_params)
        
        # transform image 2 in space of image 1
        rows, cols = img2_gray.shape
        img2_as_viewed_from_1 = cv2.warpAffine(img2_gray, M_2_to_1, (cols, rows))
        show_two_images(
            img2_gray, img2_as_viewed_from_1,
            title="Image 2 (Left) and Image 2 (Right) transformed into co-ordinates of image 1.",
        )

        # get corners for each image
        H1, W1 = img1_gray.shape
        C1 = np.array([[0, 0], [W1, 0], [0, H1], [W1, H1]])

        H2, W2 = img2_gray.shape
        C2 = np.array([[0, 0], [W2, 0], [0, H2], [W2, H2]])

        # get corners of image 2 in space of image 1
        best_params_2_to_1 = np.array(
            [
                M_2_to_1[0, 0], M_2_to_1[0, 1], M_2_to_1[1, 0],
                M_2_to_1[1, 1], M_2_to_1[0, 2], M_2_to_1[1, 2]]
        )
        C2_as_viewed_from_1 = np.dot(project_2d_to_6d(C2), best_params_2_to_1)
        C2_as_viewed_from_1 = project_1d_to_2d(C2_as_viewed_from_1)

        # get co-ordinates of image 2 transformed in image 1 space
        Y_top = np.max([0, C2_as_viewed_from_1[0, 1], C2_as_viewed_from_1[1, 1]])
        Y_bot = np.min([H1, C2_as_viewed_from_1[2, 1], C2_as_viewed_from_1[3, 1]])
        X_left = np.min([0, C2_as_viewed_from_1[0, 0], C2_as_viewed_from_1[3, 0]])
        X_right = np.max([W1, C2_as_viewed_from_1[1, 0], C2_as_viewed_from_1[2, 0]])
        H_new = Y_bot - Y_top
        W_new = X_right - X_left

        # show canvas containing both images and transformed corners
        H, W = 400, 800
        canvas = np.zeros((H, W, 3))
        canvas[:H1, :W1, :] = img1
        canvas[:H2, W - W2:, :] = img2

        # create another subcanvas containing warped version of image 2
        subcanvas = np.zeros((H, W, 3))
        subcanvas[:H2, :W2, :] = img2
        subcanvas = cv2.warpAffine(subcanvas, M_2_to_1, (W, H))

        self.show_canvas(canvas, subcanvas, C2_as_viewed_from_1, X_left, Y_top, W_new, H_new)

        # merge images with weighted sub over pixels
        merged_image = np.zeros(canvas.shape)
        merging_weights_1 = np.zeros(canvas.shape)
        merging_weights_2 = np.zeros(canvas.shape)
        checker = np.zeros(canvas.shape)

        C1_reordered = np.array([C1[0], C1[1], C1[3], C1[2]])
        P1 = Polygon(C1_reordered)

        C2_as_viewed_from_1_reordered = np.array(
            [C2_as_viewed_from_1[0], C2_as_viewed_from_1[1], C2_as_viewed_from_1[3], C2_as_viewed_from_1[2]]
        )
        P2 = Polygon(C2_as_viewed_from_1_reordered)

        for x in np.arange(int(X_left), int(X_right), 1):
            for y in np.arange(int(Y_top), int(Y_bot), 1):
                
                in_img_1 = P1.contains_point((x, y))
                in_img_2 = P2.contains_point((x, y))
                
                if in_img_1 and not in_img_2:
                    merging_weights_1[y, x] = 1.0
                    merging_weights_2[y, x] = 0.0

                if in_img_2 and not in_img_1:
                    merging_weights_1[y, x] = 0.0
                    merging_weights_2[y, x] = 1.0

                if in_img_1 and in_img_2:
                    merging_weights_1[y, x] = 0.5
                    merging_weights_2[y, x] = 0.5
                    checker[y, x] = 1.0
        
        # visualize merging weights
        fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        ax[0].axis("off")
        ax[0].imshow(merging_weights_1, cmap="gray")
        ax[0].set_title("Merging weights for image 1")
        ax[1].axis("off")
        ax[1].imshow(merging_weights_2, cmap="gray")
        ax[1].set_title("Merging weights for image 2")
        plt.show()

        # create the final stitched image
        stitched = np.multiply(canvas, merging_weights_1) + np.multiply(subcanvas, merging_weights_2)
        stitched = stitched[int(Y_top):int(Y_bot), int(X_left):int(X_right)]

        # show the stiched image
        f, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [2, 2, 4]}, figsize=(20, 6))

        ax[0].imshow(img1)
        ax[0].set_title("Image 1", fontsize=18)
        ax[0].axis("off")

        ax[1].imshow(img2)
        ax[1].set_title("Image 2", fontsize=18)
        ax[1].axis("off")

        ax[2].imshow(stitched.astype(int))
        ax[2].set_title("Stitched image", fontsize=18)
        ax[2].axis("off")

        f.tight_layout()
        os.makedirs("./results/", exist_ok=True)
        f.savefig('./results/all_stiched.png', bbox_inches="tight")
        plt.show()

        return stitched


def demo(img1_path, img2_path):
    # load both images 
    img1 = cv2.imread(img1_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    img2 = cv2.imread(img2_path)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # show input images
    show_two_images(img1, img2, title="Given pair of images.")

    # initialize stitcher
    image_stitching = ImageStitching()
    stitched = image_stitching.stitch(img1, img2)

    return stitched


if __name__ == "__main__":
    stitched = demo("left.jpg", "right.jpg")