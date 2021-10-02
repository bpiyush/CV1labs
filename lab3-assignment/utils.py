"""Helper functions"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List


def show_multiple_images(
        imgs: List[np.ndarray], grid: tuple = None, figsize=(8, 8), ax=None,
        grayscale=False, show=False, xticks=True, yticks=True, save=False, path="sample.png",
    ):
    """Displays a set of images based on given grid pattern."""
    assert isinstance(imgs, list)
    assert isinstance(grid, tuple) and len(grid) == 2

    num_imgs = len(imgs)
    if grid is None:
        grid = (1, num_imgs)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    grid_imgs = [[None for _ in range(grid[1])] for _ in range(grid[0])]
    for i in range(grid[0]):
        for j in range(grid[1]):
            grid_imgs[i][j] = imgs[i * j + j]

    disp_imgs = []
    for i in range(grid[0]):
        disp_imgs.append(np.hstack(grid_imgs[i]))
    disp_imgs = np.vstack(disp_imgs)
    
    args = {"X": disp_imgs}
    if grayscale:
        args["cmap"] = "gray"

    ax.imshow(**args)

    if not xticks:
        ax.set_xticks([])
    
    if not yticks:
        ax.set_yticks([])
    
    if save:
        assert isinstance(path, str)
        plt.savefig(path, bbox_inches="tight")

    if show:
        plt.show()
