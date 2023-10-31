import sys
sys.path.append('./rt_erg_lib/')
from double_integrator import DoubleIntegrator
from ergodic_control import RTErgodicControl
from target_dist import TargetDist
from target_dist_t import TargetDist_t

from utils import convert_phi2phik, convert_ck2dist, convert_traj2ck, convert_phik2phi
import numpy as np
from scipy.io import loadmat

import matplotlib.pyplot as plt

import os

if __name__ == '__main__':
    env         = DoubleIntegrator() # robot controller system
    model       = DoubleIntegrator()
    if (0):
        t_dist      = TargetDist(10)
    else:
        data = loadmat('/home/clp/catkin_ws/src/ada_sampling/scripts/ada_sampling/ship_trajectory_old_40_20.mat')
        distance_map = data['F_map']
        t_dist      = TargetDist_t(distance_map)
    
    erg_ctrl    = RTErgodicControl(model, horizon=15, num_basis=10, batch_size=-1)
    
    
    # phi -> phik
    erg_ctrl.phik = convert_phi2phik(erg_ctrl.basis, t_dist.grid_vals, t_dist.grid)
    # convert back (the whole pic)
    # phik -> phi
    phi = convert_phik2phi(erg_ctrl.basis, erg_ctrl.phik, t_dist.grid)
    plt.imshow(phi.reshape(40,20), cmap='viridis')
    plt.colorbar()
    plt.show()
    print(np.mean(phi))
    
    
    reshaped_matrix = t_dist.grid_vals.reshape((40, 20))
    # original matrix
    # plt.imshow(reshaped_matrix, cmap='viridis')
    # plt.colorbar()
    # plt.show()
    
    # 创建上半部分的矩阵副本，并将下半部分的值设置为零
    upper_half_matrix = reshaped_matrix.copy()
    # upper_half_matrix[20:, :] = 0
    phik_up = convert_phi2phik(erg_ctrl.basis, upper_half_matrix.reshape(-1,1), t_dist.grid)

    # 创建下半部分的矩阵副本，并将上半部分的值设置为零
    lower_half_matrix = reshaped_matrix.copy()
    # lower_half_matrix[:20, :] = 0
    phik_low = convert_phi2phik(erg_ctrl.basis, lower_half_matrix.reshape(-1,1), t_dist.grid)
    
    result = (phik_up + phik_low) 

    phi = convert_phik2phi(erg_ctrl.basis, result, t_dist.grid)
    print(np.mean(phi))
    plt.imshow(phi.reshape(40,20), cmap='viridis')
    plt.colorbar()
    plt.show()
    # 定义一个函数来绘制矩阵
    def plot_matrix(matrix, title):
        plt.figure()
        plt.imshow(matrix, cmap='viridis')
        plt.colorbar()
        plt.title(title)
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.show()

    # # 绘制上半部分
    # plot_matrix(upper_half_matrix, "Upper Half of the Matrix")

    # # 绘制下半部分
    # plot_matrix(lower_half_matrix, "Lower Half of the Matrix")





    
    # # setting the phik on the ergodic controller
    # erg_ctrl.phik = convert_phi2phik(erg_ctrl.basis, t_dist.grid_vals, t_dist.grid)
    # plt.colorbar()
   
    # plt.show()
