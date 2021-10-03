"""Script to implement feature tracking."""
from os.path import join, exists
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob

from lucas_kanade import lucas_kanade, lucas_kanade_for_points
from harris_corner_detector import harris_corner_detector, show_derivatives_and_corners
from utils import make_video


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
    
    # remove points that go outside the image
    H, W = I0.shape
    indices = np.where(np.add(points[:, 0] > H, points[:, 1] > W))[0]
    points = np.delete(points, indices, axis=0)

    images_with_t = [I0.copy()]
    points_with_t = [points.copy()]
    V_with_t = [None]

    for i in range(1, len(impaths)):

        # collect current image and features
        I_curr = images_with_t[-1]
        points_curr = points_with_t[-1]

        # read the next image
        I_next = read_image(impaths[i], convert=cv2.COLOR_BGR2GRAY)

        # compute optical flow
        V = lucas_kanade_for_points(I_curr, I_next, points_curr, make_plot=False)
        V_with_t.append(V)

        # get points for the new image ((x, y) + (Vx, Vy))
        H, W = I_curr.shape
        points_next = points_curr.copy()
        points_next[:, 0] += (V[:, 1] * 5).astype(int)
        points_next[:, 1] += (V[:, 0] * 5).astype(int)

        # remove points that go outside the image
        indices = np.where(np.add(points_next[:, 0] > H, points_next[:, 1] > W))[0]
        points_next = np.delete(points_next, indices, axis=0)

        # add to our tracker
        points_with_t.append(points_next)
        images_with_t.append(I_next)

    # make a video
    video_frames = []

    fig = plt.figure()
    plt.imshow(images_with_t[0])
    plt.scatter(points_with_t[0][:, 1], points_with_t[0][:, 0], c='red', marker='.')
    plt.draw()
    plt.axis("off")
    plt.pause(0.0001)
    img_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_plot = np.reshape(img_plot, fig.canvas.get_width_height()[::-1] + (3,))
    img_plot = cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB)
    video_frames.append(img_plot)

    for i in range(1, len(images_with_t)):
        image = images_with_t[i]
        prev_points = points_with_t[i - 1]
        points = points_with_t[i]
        V = V_with_t[i]

        # Clears the entire current figure with all its axes, but leaves the window.
        plt.clf()
        plt.imshow(image)
        # plt.scatter(points[:, 1], points[:, 0], c='red', marker='.')
        plt.quiver(points[:, 1], points[:, 0], V[:, 0], V[:, 1], angles='xy', scale_units='xy', scale=0.1)
        plt.draw()
        plt.axis("off")
        plt.pause(0.0001)
        img_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_plot = np.reshape(img_plot, fig.canvas.get_width_height()[::-1]+(3,))
        img_plot = cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB)
        video_frames.append(img_plot)
    
    path = f"results/{example}_flow.avi"
    make_video(video_frames, path=path, convert_to_rgb=False)

