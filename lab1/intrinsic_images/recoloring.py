"""Script to recolor ball.png"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

from iid_image_formation import form_image_from_decompositions


if __name__ == "__main__":
    oimg = cv2.imread("./ball.png")
    oimg = cv2.cvtColor(oimg, cv2.COLOR_BGR2RGB)

    albedo = cv2.imread("./ball_albedo.png")
    albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)

    shading = cv2.imread("./ball_shading.png")
    shading = cv2.cvtColor(shading, cv2.COLOR_BGR2RGB)

    # find true color of the ball
    colors_in_albedo = set(tuple(v) for point_2d in albedo for v in point_2d)
    assert len(colors_in_albedo) == 2, "more than 2 colors found in the albedo"
    assert (0, 0, 0) in colors_in_albedo, "background color in albedo not black"
    # remove background color
    colors_in_albedo.remove((0, 0, 0))
    # extract the remaining color as the ball color
    ball_color = list(colors_in_albedo)[0]
    ball_color = list(ball_color)
    print(f"True color of the ball (in RGB space): {ball_color}")

    # replace the ball color to be green
    new_albedo = albedo.copy()
    new_albedo[albedo[:, :, 0] == ball_color[0], 0] = 0
    new_albedo[albedo[:, :, 1] == ball_color[1], 1] = 255
    new_albedo[albedo[:, :, 2] == ball_color[2], 2] = 0

    # recolor the ball
    nimg = form_image_from_decompositions(new_albedo, shading)

    # show the images
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(oimg)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_xlabel("Original image", fontsize=12)

    ax[1].imshow(nimg)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_xlabel("Recolored image", fontsize=12)

    plt.savefig("./results/recolored.png", bbox_inches="tight")
    plt.show()


