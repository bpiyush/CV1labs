import os
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List

def load_syn_images(image_dir='./SphereGray5/', channel=0):
    files = os.listdir(image_dir)
    #files = [os.path.join(image_dir, f) for f in files]
    nfiles = len(files)
    
    image_stack = None
    V = 0
    Z = 0.5
    
    for i in range(nfiles):
        # read input image
        im = cv2.imread(os.path.join(image_dir, files[i]))
        im = im[:,:,channel]
        
        # stack at third dimension
        if image_stack is None:
            h, w = im.shape
            print('Image size (H*W): %d*%d' %(h,w) )
            image_stack = np.zeros([h, w, nfiles], dtype=int)
            V = np.zeros([nfiles, 3], dtype=np.float64)
            
        image_stack[:,:,i] = im
        
        # read light direction from image name
        X = np.double(files[i][(files[i].find('_')+1):files[i].rfind('_')])
        Y = np.double(files[i][files[i].rfind('_')+1:files[i].rfind('.png')])
        V[i, :] = [-X, Y, Z]
        
    # normalization
    image_stack = np.double(image_stack)
    min_val = np.min(image_stack)
    max_val = np.max(image_stack)
    image_stack = (image_stack - min_val) / (max_val - min_val)
    normV = np.tile(np.sqrt(np.sum(V ** 2, axis=1, keepdims=True)), (1, V.shape[1]))
    scriptV = V / normV
    
    return image_stack, scriptV
    
    
def load_face_images(image_dir='./yaleB02/'):
    num_images = 64
    filename = os.path.join(image_dir, 'yaleB02_P00_Ambient.pgm')
    ambient_image = cv2.imread(filename, -1)
    h, w = ambient_image.shape

    # get list of all other image files
    import glob 
    d = glob.glob(os.path.join(image_dir, 'yaleB02_P00A*.pgm'))
    import random
    d = random.sample(d, num_images)
    filenames = [os.path.basename(x) for x in d]

    ang = np.zeros([2, num_images])
    image_stack = np.zeros([h, w, num_images])

    for j in range(num_images):
        ang[0,j], ang[1,j] = np.double(filenames[j][12:16]), np.double(filenames[j][17:20])
        image_stack[...,j] = cv2.imread(os.path.join(image_dir, filenames[j]), -1) - ambient_image


    x = np.cos(np.pi*ang[1,:]/180) * np.cos(np.pi*ang[0,:]/180)
    y = np.cos(np.pi*ang[1,:]/180) * np.sin(np.pi*ang[0,:]/180)
    z = np.sin(np.pi*ang[1,:]/180)
    scriptV = np.array([y,z,x]).transpose(1,0)

    image_stack = np.double(image_stack)
    image_stack[image_stack<0] = 0
    min_val = np.min(image_stack)
    max_val = np.max(image_stack)
    image_stack = (image_stack - min_val) / (max_val - min_val)
    
    return image_stack, scriptV
    
    
def show_results(albedo, normals, height_map, SE):
    # Stride in the plot, you may want to adjust it to different images
    stride = 1
    
    # showing albedo map
    fig = plt.figure()
    albedo_max = albedo.max()
    albedo_max = 1
    albedo = albedo / albedo_max
    print(albedo.shape)
    plt.imshow(albedo, cmap="gray")
    plt.show()
    
    # showing normals as three separate channels
    figure = plt.figure()
    ax1 = figure.add_subplot(131)
    ax1.imshow(normals[..., 0])
    ax2 = figure.add_subplot(132)
    ax2.imshow(normals[..., 1])
    ax3 = figure.add_subplot(133)
    ax3.imshow(normals[..., 2])
    plt.show()
    
    # meshgrid
    X, Y, _ = np.meshgrid(np.arange(0,np.shape(normals)[0], stride),
    np.arange(0,np.shape(normals)[1], stride),
    np.arange(1))
    X = X[..., 0]
    Y = Y[..., 0]
    
    '''
    =============
    You could further inspect the shape of the objects and normal directions by using plt.quiver() function.  
    =============
    '''
    
    # plotting the SE
    H = SE[::stride,::stride]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X,Y, H.T)
    plt.show()
    
    # plotting model geometry
    H = height_map[::stride,::stride]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X,Y, H.T)
    plt.show()


def convert_polar_to_cartesian(theta: np.ndarray, phi: np.ndarray, convert_to_rad=False):
    """Converts (theta, phi) in polar coordinates to cartesian coordinates"""
    # convert in radians
    if convert_to_rad:
        theta = (np.pi * theta) / 180.0
        phi = (np.pi * phi) / 180.0

    # compute x, y, z
    z = np.cos(theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)

    return np.array([x, y, z]).T


def show_single_image(
        img: np.ndarray, figsize=(8, 8), ax=None, show=True, grayscale=False,
        xticks=True, yticks=True, save=False, path="sample.png",
    ):
    """Displays a single image."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    args = {"X": img}
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