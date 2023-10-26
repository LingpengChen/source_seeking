import numpy as np


def convert_phi2phik(basis, phi_val, phi_grid=None):
    '''
    Converts the distribution to the fourier decompositions
    '''
    if len(phi_val.shape) != 1:
        phi_val = phi_val.ravel()
    if phi_grid is None:
        print('--Assuming square grid')
        phi_grid = np.meshgrid(*[np.linspace(0, 1., int(np.sqrt(phi_val)))
                                for _ in range(2)])

    assert phi_grid.shape[0] == phi_val.shape[0], 'samples are not the same'

    res = np.sum([basis.fk(x) * v for v, x in zip(phi_val, phi_grid)], axis=0)
    return res

def convert_phik2phi(basis, phik, phi_grid=None):
    '''
    Reconstructs phi from the Fourier terms
    '''
    if phi_grid is None:
        print('--Assuming square grid')
        phi_grid = np.meshgrid(*[np.linspace(0, 1.)
                                for _ in range(2)])
        phi_grid = np.c_[phi_grid[0].ravel(), phi_grid[1].ravel()]
    phi_val = np.stack([np.dot(basis.fk(x), phik) for x in phi_grid])
    return phi_val

def convert_traj2ck(basis, xt):
    '''
    This utility function converts a trajectory into its time-averaged  
    statistics in the Fourier domain
    '''
    # function (3)  xt are path points
    N = len(xt)
    return np.sum([basis.fk(x) for x in xt], axis=0) / N

def convert_ck2dist(basis, ck, grid=None):
    '''
    This utility function converts a ck into its time-averaged
    statistics
    '''
    if grid is None:
        print('--Assuming square grid')
        grid = np.meshgrid(*[np.linspace(0, 1.)
                                for _ in range(2)])
        grid = np.c_[grid[0].ravel(), grid[1].ravel()]

    val = np.stack([np.dot(basis.fk(x), ck) for x in grid])
    return val

from scipy.ndimage import maximum_filter

def find_peak(matrix):
    
    # 使用最大滤波器找到每个位置的局部最大值
    local_max = maximum_filter(matrix, size=3) == matrix
    
    # 获取局部最大值的坐标
    local_maxima_coords = np.argwhere(local_max)
    
    # 过滤出严格大于其周围邻居的局部最大值
    strict_local_maxima = []
    for i, j in local_maxima_coords:
        if i > 0 and j > 0 and i < matrix.shape[0] - 1 and j < matrix.shape[1] - 1:
            neighbors = [matrix[i-1, j-1], matrix[i-1, j], matrix[i-1, j+1],
                         matrix[i, j-1],                 matrix[i, j+1],
                         matrix[i+1, j-1], matrix[i+1, j], matrix[i+1, j+1]]
            if all(matrix[i, j] > neighbor for neighbor in neighbors):
                strict_local_maxima.append((i,j))
    
    return strict_local_maxima

# def find_peak_auto(gp, resolution=50):
    
#     # 使用最大滤波器找到每个位置的局部最大值
#     local_max = maximum_filter(matrix, size=3) == matrix
    
#     # 获取局部最大值的坐标
#     local_maxima_coords = np.argwhere(local_max)
    
#     # 过滤出严格大于其周围邻居的局部最大值
#     strict_local_maxima = []
#     for i, j in local_maxima_coords:
#         if i > 0 and j > 0 and i < matrix.shape[0] - 1 and j < matrix.shape[1] - 1:
#             neighbors = [matrix[i-1, j-1], matrix[i-1, j], matrix[i-1, j+1],
#                          matrix[i, j-1],                 matrix[i, j+1],
#                          matrix[i+1, j-1], matrix[i+1, j], matrix[i+1, j+1]]
#             if all(matrix[i, j] > neighbor for neighbor in neighbors):
#                 strict_local_maxima.append((i,j))
    
#     return strict_local_maxima

# # 示例
# matrix = [
#     [1, 2, 3, 2, 1],
#     [5, 3, 4, 4, 2],
#     [3, 7, 1, 8, 3],
#     [4, 2, 3, 5, 2],
#     [2, 3, 4, 2, 1]
# ]
# print(find_local_maxima_2d(matrix))  # 输出: [[1, 0], [2, 1], [2, 3], [3, 3]]
