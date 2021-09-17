import numpy as np

def estimate_alb_nrm( image_stack, scriptV, shadow_trick=True):
    
    # COMPUTE_SURFACE_GRADIENT compute the gradient of the surface
    # INPUT:
    # image_stack : the images of the desired surface stacked up on the 3rd dimension
    # scriptV : matrix V (in the algorithm) of source and camera information
    # shadow_trick: (true/false) whether or not to use shadow trick in solving linear equations
    # OUTPUT:
    # albedo : the surface albedo
    # normal : the surface normal

    h, w, _ = image_stack.shape
    
    # create arrays for 
    # albedo (1 channel)
    # normal (3 channels)
    albedo = np.zeros([h, w])
    normal = np.zeros([h, w, 3])
    
    """
    ================
    Your code here
    ================
    for each point in the image array
        stack image values into a vector i
        construct the diagonal matrix scriptI
        solve scriptI * scriptV * g = scriptI * i to obtain g for this point
        albedo at this point is |g|
        normal at this point is g / |g|
    """

    for y in range(h):
        for x in range(w):
            i = image_stack[y, x, :]
            scriptI = np.diag(i)

            if shadow_trick:
                g = np.linalg.lstsq(a=scriptI @ scriptV, b=np.dot(scriptI, i), rcond=None)[0]
            else:
                g = np.linalg.lstsq(a=scriptV, b=i, rcond=None)[0]

            albedo[y, x] = np.linalg.norm(g)
            normal[y, x] = (g / (np.linalg.norm(g) + np.finfo(float).eps))
    
    return albedo, normal
    
if __name__ == '__main__':
    n = 5
    image_stack = np.zeros([10,10,n])
    scriptV = np.zeros([n,3])
    estimate_alb_nrm( image_stack, scriptV, shadow_trick=True)