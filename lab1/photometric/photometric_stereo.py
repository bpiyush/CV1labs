import numpy as np
import cv2
import os
from utils import *
from estimate_alb_nrm import estimate_alb_nrm
from check_integrability import check_integrability
from construct_surface import construct_surface

print('Part 1: Photometric Stereo\n')

def photometric_stereo(image_dir='./SphereGray5/', files=None, shadow_trick=True, show=True, return_cache=False, channel=0):

    # obtain many images in a fixed view under different illumination
    print('Loading images...\n')
    [image_stack, scriptV] = load_syn_images(image_dir=image_dir, files=files, channel=channel)
    [h, w, n] = image_stack.shape
    print('Finish loading %d images.\n' % n)

    # compute the surface gradient from the stack of imgs and light source mat
    print('Computing surface albedo and normal map...\n')
    [albedo, normals] = estimate_alb_nrm(image_stack, scriptV, shadow_trick=shadow_trick)


    # integrability check: is (dp / dy  -  dq / dx) ^ 2 small everywhere?
    print('Integrability checking\n')
    [p, q, SE] = check_integrability(normals)

    threshold = 0.005;
    print('Number of outliers: %d\n' % np.sum(SE > threshold))
    SE[SE <= threshold] = float('nan') # for good visualization

    # compute the surface height
    height_map = construct_surface( p, q, path_type="column" )

    # show results
    if show:
        show_results(albedo, normals, height_map, SE)

    if return_cache:
        return albedo, normals, height_map, SE


# Color images
def photometric_stereo_color(image_dir='./SphereColor/', num_channels=3, files=None, shadow_trick=False):

    albedos = []
    normals = []
    height_maps = []

    for c in range(num_channels):
        albedo, normal, height_map, SE = photometric_stereo(
            image_dir, channel=c, show=False, return_cache=True, shadow_trick=shadow_trick
        )
        albedos.append(albedo)
        normals.append(normal)
        height_maps.append(height_map)

    return albedos, normals, height_maps

## Face
def photometric_stereo_face(image_dir='./yaleB02/', path_type="column", show=True, return_cache=False):
    [image_stack, scriptV] = load_face_images(image_dir)
    [h, w, n] = image_stack.shape
    print('Finish loading %d images.\n' % n)
    print('Computing surface albedo and normal map...\n')
    albedo, normals = estimate_alb_nrm(image_stack, scriptV, shadow_trick=False)

    # integrability check: is (dp / dy  -  dq / dx) ^ 2 small everywhere?
    print('Integrability checking')
    p, q, SE = check_integrability(normals)

    threshold = 0.005;
    print('Number of outliers: %d\n' % np.sum(SE > threshold))
    SE[SE <= threshold] = float('nan') # for good visualization

    # compute the surface height
    height_map = construct_surface( p, q, path_type=path_type)

    # show results
    if show:
        show_results(albedo, normals, height_map, SE, set_lim=False)

    if return_cache:
        return albedo, normals, height_map, SE
    
if __name__ == '__main__':
    # photometric_stereo('./SphereGray5/')
    # photometric_stereo('./MonkeyGray/', shadow_trick=True)

    # photometric_stereo('./SphereColor/', shadow_trick=False)
    # photometric_stereo('./MonkeyColor/', shadow_trick=False)

    photometric_stereo_face(path_type="column")
    # photometric_stereo_face(path_type="row")
    # photometric_stereo_face(path_type="average")

    # photometric_stereo_color('./SphereColor/')

    # for multi-channel outputs
    # albedos, normals, height_maps = photometric_stereo_color("./SphereColor/")
    # albedos, normals, height_maps = photometric_stereo_color("./MonkeyColor/")
