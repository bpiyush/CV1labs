import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


def check_extreme(arr):
    for i in range(arr.shape[-1]):
        print(f"Axis {i} - Min: {arr[:, :, i].min()} Max: {arr[:, :, i].max()}")


def imshow_helper(img, ax, title, xticks=False, yticks=False):
    ax.imshow(img)
    ax.set_title(title)
    if not xticks:
        ax.set_xticks([])
    if not yticks:
        ax.set_yticks([])


def opponent2rgb(input_image):
    # converts an image from opponent colour space to RGB space
    matrix = np.array(
        [
            [1.0 / np.sqrt(2), -1.0 / np.sqrt(2), 0],
            [1.0 / np.sqrt(6), 1.0 / np.sqrt(6), -2.0 / np.sqrt(6)],
            [1.0 / np.sqrt(3), 1.0 / np.sqrt(3), 1.0 / np.sqrt(3)]
        ]
    )
    new_image = np.dot(input_image, np.linalg.inv(matrix))
    new_image = (new_image * 255).astype("uint8")

    return new_image


def visualize(input_image, img_type):
    # Fill in this function. Remember to remove the pass command
    fig, ax = plt.subplots(1, 4, figsize=(10, 4))
    plt.hsv()

    img_type_to_channels = {
        "opponent": ["O1", "O2", "O3"],
        "rgb": ["r", "g", "b"],
        "ycbcr": ["Y", "Cb", "Cr"],
        "hsv": ["Hue", "Saturation", "Value"],
        "original": ["R", "G", "B"],
    }

    if img_type == "original":
        imshow_helper(input_image, ax[0], "Complete image")
        for i in range(3):
            channel = np.zeros((input_image.shape))
            channel[:, :, i] = input_image[:, :, i]
            imshow_helper(
                channel.astype("uint8"), ax[i + 1],
                "Channel: {}".format(img_type_to_channels["original"][i])
            )

    elif img_type == "gray":
        titles = ["Lightness", "Average", "Luminosity", "OpenCV"]
        assert input_image.shape[-1] == 4
        for i in range(input_image.shape[-1]):
            ax[i].imshow(input_image[..., i], cmap="gray")
            ax[i].set_title(titles[i])
            ax[i].set_xticks([])
            ax[i].set_yticks([])

    elif img_type == "hsv":
        # set complete image
        x = cv2.cvtColor(input_image.astype("float32"), cv2.COLOR_HSV2RGB)
        imshow_helper(x, ax[0], "Complete image")

        # show H
        H = np.ones(input_image.shape)
        H[:, :, 0] = input_image[:, :, 0]
        x = cv2.cvtColor(H.astype("float32"), cv2.COLOR_HSV2RGB)
        imshow_helper(x, ax[1], "Channel: Hue")

        # show S
        S = np.ones(input_image.shape)
        S[:, :, 1] = input_image[:, :, 1]
        x = cv2.cvtColor(S.astype("float32"), cv2.COLOR_HSV2RGB)
        imshow_helper(x, ax[2], "Channel: Saturation")

        # show V
        V = np.ones(input_image.shape)
        V[:, :, 2] = input_image[:, :, 1]
        V[:, :, 1] = 0.0
        x = cv2.cvtColor(V.astype("float32"), cv2.COLOR_HSV2RGB)
        imshow_helper(x, ax[3], "Channel: Value")
    
    elif img_type == "ycbcr":
        # set complete image
        x = cv2.cvtColor(input_image.astype("float32"), cv2.COLOR_YCrCb2RGB)
        imshow_helper((x * 255).astype("uint8"), ax[0], "Complete image")

        # show Y
        Y = np.zeros(input_image.shape)
        Y[:, :, 0] = input_image[:, :, 0]
        x = cv2.cvtColor(Y.astype("float32"), cv2.COLOR_YCrCb2RGB)
        imshow_helper((x * 255).astype("uint8"), ax[1], "Channel: Y")

        # show Cr
        Cr = np.zeros(input_image.shape)
        Cr[:, :, 1] = input_image[:, :, 1]
        x = cv2.cvtColor(Cr.astype("float32"), cv2.COLOR_YCrCb2RGB)
        imshow_helper((x * 255).astype("uint8"), ax[2], "Channel: Cr")

        # show Cb
        Cb = np.zeros(input_image.shape)
        Cb[:, :, 2] = input_image[:, :, 2]
        x = cv2.cvtColor(Cb.astype("float32"), cv2.COLOR_YCrCb2RGB)
        imshow_helper((x * 255).astype("uint8"), ax[3], "Channel: Cb")
    
    elif img_type == "opponent":
        # set complete image
        x = opponent2rgb(input_image)
        imshow_helper(x, ax[0], "Complete image")

        # show O1
        O1 = np.zeros(input_image.shape)
        O1[:, :, 0] = input_image[:, :, 0]
        x = opponent2rgb(O1)
        imshow_helper(x, ax[1], "Channel: $O_{1}$")

        # show O2
        O2 = np.zeros(input_image.shape)
        O2[:, :, 1] = input_image[:, :, 1]
        x = opponent2rgb(O2)
        imshow_helper(x, ax[2], "Channel: $O_{2}$")

        # show O3
        O3 = np.zeros(input_image.shape)
        O3[:, :, 2] = input_image[:, :, 2]
        x = opponent2rgb(O3)
        imshow_helper(x, ax[3], "Channel: $O_{3}$")

    else:
        # set complete image
        ax[0].imshow(input_image)
        ax[0].set_title("Complete image")
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        # set individual channels
        for i in range(3):
            ax[i + 1].imshow(input_image[:, :, i])
            ax[i + 1].set_title("Channel {}".format(img_type_to_channels[img_type][i]))
            ax[i + 1].set_xticks([])
            ax[i + 1].set_yticks([])

    fpath = "./{}.png".format(img_type)
    plt.savefig(fpath, bbox_inches="tight")

    plt.show()
