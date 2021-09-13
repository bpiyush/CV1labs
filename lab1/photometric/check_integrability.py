import numpy as np

def check_integrability(normals):
    #  CHECK_INTEGRABILITY check the surface gradient is acceptable
    #   normals: normal image
    #   p : df / dx
    #   q : df / dy
    #   SE : Squared Errors of the 2 second derivatives

    # initalization
    p = np.zeros(normals.shape[:2])
    q = np.zeros(normals.shape[:2])
    SE = np.zeros(normals.shape[:2])
    
    """
    ================
    Your code here
    ================
    Compute p and q, where
    p measures value of df / dx
    q measures value of df / dy
    
    """
    h, w = p.shape
    for y in range(h):
        for x in range(w):
            p[y, x] = (normals[y, x, 0] / (normals[y, x, -1] + np.finfo(float).eps))
            q[y, x] = (normals[y, x, 1] / (normals[y, x, -1] + np.finfo(float).eps))
    
    # change nan to 0
    p[p!=p] = 0
    q[q!=q] = 0
    
    """
    ================
    Your code here
    ================
    approximate second derivate by neighbor difference
    and compute the Squared Errors SE of the 2 second derivatives SE
    
    """
    # compute \del p / \del y
    del_p_by_del_y = np.diff(p, axis=1, prepend=0)

    # compute \del q / \del x
    del_q_by_del_x = np.diff(q, axis=0, prepend=0)

    SE = (del_p_by_del_y - del_q_by_del_x) ** 2

    return p, q, SE


if __name__ == '__main__':
    normals = np.zeros([10,10,3])
    check_integrability(normals)