import sys
sys.path.append('./rt_erg_lib/')
from double_integrator import DoubleIntegrator
from ergodic_control import RTErgodicControl
from target_dist import TargetDist
from target_dist_t import TargetDist_t

from environment_and_measurement import f

from utils import convert_phi2phik, convert_ck2dist, convert_traj2ck, convert_phik2phi
import numpy as np
from scipy.io import loadmat

import matplotlib.pyplot as plt

import os

if __name__ == '__main__':
    env         = DoubleIntegrator() # robot controller system
    model       = DoubleIntegrator()
    # if (0):
    #     t_dist      = TargetDist(10)
    # else:
    #     data = loadmat('/home/clp/catkin_ws/src/ada_sampling/scripts/ada_sampling/ship_trajectory_old_40_20.mat')
    #     distance_map = data['F_map']
    #     t_dist      = TargetDist_t(distance_map)
    
    x_min = (0, 0)
    x_max = (10, 10)
    test_resolution = [50, 50]
    X_test_x = np.linspace(x_min[0], x_max[0], test_resolution[0])
    X_test_y = np.linspace(x_min[1], x_max[1], test_resolution[1])
    X_test_xx, X_test_yy = np.meshgrid(X_test_x, X_test_y)
    grid_2_r_w = np.meshgrid(np.linspace(0, 1, int(test_resolution[0])), np.linspace(0, 1, int(test_resolution[1])))
    grid_vals = f(np.vstack(np.dstack((X_test_xx, X_test_yy)))).reshape(50,50)
    grid = np.c_[grid_2_r_w[0].ravel(), grid_2_r_w[1].ravel()] #(2500,2)
    
    # original normalized image
    grid_vals /= np.sum(grid_vals)
    print("Original:",np.sum(grid_vals))
    
    plt.imshow(grid_vals, cmap='viridis')
    plt.colorbar()
    plt.show()
    
    erg_ctrl    = RTErgodicControl(model, horizon=15, num_basis=10, batch_size=-1)
    
    ######## 
    # test = np.ones(test_resolution)
    # test /= np.sum(test)
    # phik = convert_phi2phik(erg_ctrl.basis, test, grid)
    # print(np.mean(phik)/3) 
    # phi -> phik
    phik = convert_phi2phik(erg_ctrl.basis, grid_vals, grid)
    print("first phik", np.mean(phik))   
    # phik -> phi
    phi = convert_phik2phi(erg_ctrl.basis, phik, grid)
    
    print("Changed back to sum", np.sum(phi))   
    
    plt.imshow(phi.reshape(50,50), cmap='viridis')
    plt.colorbar()
    plt.show()
    
    
    reshaped_matrix = grid_vals.reshape((50,50))
    
    # 创建上半部分的矩阵副本，并将下半部分的值设置为零
    upper_half_matrix = reshaped_matrix.copy()
    # upper_half_matrix[20:, :] = 0
    phik_up = convert_phi2phik(erg_ctrl.basis, upper_half_matrix.reshape(-1,1), grid)
    print(np.mean(phik_up))    # convert back (the whole pic)

    # 创建下半部分的矩阵副本，并将上半部分的值设置为零
    lower_half_matrix = reshaped_matrix.copy()
    # lower_half_matrix[:20, :] = 0
    phik_low = convert_phi2phik(erg_ctrl.basis, lower_half_matrix.reshape(-1,1), grid)
    print(np.mean(phik_low))    # convert back (the whole pic)
    
    result = (phik_up + phik_low) 

    phi = convert_phik2phi(erg_ctrl.basis, result, grid)

    plt.imshow(phi.reshape(50,50), cmap='viridis')
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
