import numpy as np

def construct_surface(p, q, path_type='column'):

    '''
    CONSTRUCT_SURFACE construct the surface function represented as height_map
       p : measures value of df / dx
       q : measures value of df / dy
       path_type: type of path to construct height_map, either 'column',
       'row', or 'average'
       height_map: the reconstructed surface
    '''
    
    h, w = p.shape
    height_map = np.zeros([h, w])
    
    if path_type=='column':
        """
        ================
        Your code here
        ================
        % top left corner of height_map is zero
        % for each pixel in the left column of height_map
        %   height_value = previous_height_value + corresponding_q_value
        
        % for each row
        %   for each element of the row except for leftmost
        %       height_value = previous_height_value + corresponding_p_value
        
        """
        height_map[0, 0] = 0.0
        for y in range(1, h):
            height_map[y, 0] = height_map[y - 1, 0] + q[y, 0]
        
        for y in range(h):
            for x in range(1, w):
                height_map[y, x] = height_map[y, x - 1] + p[y, x]

    elif path_type=='row':
        """
        ================
        Your code here
        ================
        """
        height_map[0, 0] = 0.0
        for x in range(1, w):
            height_map[0, x] = height_map[0, x - 1] + p[0, x]
        
        for x in range(w):
            for y in range(1, h):
                height_map[y, x] = height_map[y - 1, x] + q[y, x]

    elif path_type=='average':
        """
        ================
        Your code here
        ================
        """
        height_map_col = np.zeros([h, w])
        height_map_row = np.zeros([h, w])

        # compute col-wise
        height_map_col[0, 0] = 0.0
        for y in range(1, h):
            height_map_col[y, 0] = height_map_col[y - 1, 0] + q[y, 0]
        for y in range(h):
            for x in range(1, w):
                height_map_col[y, x] = height_map_col[y, x - 1] + p[y, x]

        # compute row-wise
        height_map_row[0, 0] = 0.0
        for x in range(1, w):
            height_map_row[0, x] = height_map_row[0, x - 1] + p[0, x]
        for x in range(w):
            for y in range(1, h):
                height_map_row[y, x] = height_map_row[y - 1, x] + q[y, x]

        height_map = (height_map_col + height_map_row) / 2.0
        
    return height_map
        

if __name__ == "__main__":
    p = np.random.randn(10, 10)
    q = np.random.randn(10, 10)

    height_map = construct_surface(p, q, "column")
    assert height_map.shape == (10, 10)

    height_map = construct_surface(p, q, "row")
    assert height_map.shape == (10, 10)

    height_map = construct_surface(p, q, "average")
    assert height_map.shape == (10, 10)