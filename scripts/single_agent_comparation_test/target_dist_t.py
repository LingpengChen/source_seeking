import numpy as np
import numpy.random as npr
from scipy.io import loadmat
import matplotlib.pyplot as plt

class TargetDist_t(object):
    '''
    This is going to be a test template for the code,
    eventually a newer version will be made to interface with the
    unity env
    '''

    def __init__(self, map):
        self.map = map
        # self.map[:, :15] = 0
        self.row = map.shape[0] # 40
        self.col = map.shape[1] # 20
        print(self.row, self.col)
        self.grid_2_r_w = np.meshgrid(np.linspace(0, 1, int(self.col)), np.linspace(0, 1, int(self.row)))
        self.grid = np.c_[self.grid_2_r_w[0].ravel(), self.grid_2_r_w[1].ravel()]
        
        self.grid_vals = self.__call__()
    
    def get_grid_spec(self):
        return self.grid_2_r_w, self.map

    def __call__(self):
        
        val = self.map.reshape(-1, 1)
        val /= np.sum(val)
        return val

if __name__ == '__main__':

    data = loadmat('ship_trajectory.mat')
    variable1 = data['map_length']
    variable1 = data['map_width']
    distance_map = data['F_map']
    targets = data['targets']

    t_dist      = TargetDist(distance_map)

    
    # matrix = np.array([[0, 1, 2], [3, 4, 5]])
    # t_dist      = TargetDist(matrix)
    xy, vals = t_dist.get_grid_spec()


    # xy = np.array(xy)
    # vals = np.array(vals)
    # print(xy.shape)
    # print(vals.shape)
    plt.figure(1)
    plt.title('ergodic coverage')
    plt.contourf(*xy, vals, levels=10)
    plt.show()
    distance_map = np.zeros((4, 2))  # 这里仅用全零矩阵作为示例

    # 获取 x 轴和 y 轴的坐标矩阵
    

    # # 输出结果
    # print("x 轴坐标矩阵:")
    # print(x_coords)
    # print("\ny 轴坐标矩阵:")
    # print(y_coords)
