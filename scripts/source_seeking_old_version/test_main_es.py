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
# from environment_and_measurement import f

import os

f = lambda x: source1.pdf(x) + 1.05*source2.pdf(x) + source3.pdf(x) + 1.1*source4.pdf(x) + source5.pdf(x)
if __name__ == '__main__':
    env         = DoubleIntegrator() # robot controller system
    model       = DoubleIntegrator()
    x_min = (0, 0)
    x_max = (10, 10)
    test_resolution = [50, 50]
    X_test_x = np.linspace(x_min[0], x_max[0], test_resolution[0])
    X_test_y = np.linspace(x_min[1], x_max[1], test_resolution[1])
    X_test_xx, X_test_yy = np.meshgrid(X_test_x, X_test_y)
    grid_2_r_w = np.meshgrid(np.linspace(0, 1, int(test_resolution[0])), np.linspace(0, 1, int(test_resolution[1])))
    # grid_vals = f(np.vstack(np.dstack((X_test_xx, X_test_yy)))).reshape(50,50)
    grid_vals = 10*np.ones((50,50))
    grid = np.c_[grid_2_r_w[0].ravel(), grid_2_r_w[1].ravel()] #(2500,2)
    
    # original normalized image
    grid_vals /= np.sum(grid_vals)
    print("Original:",np.sum(grid_vals))
    
    # plt.imshow(grid_vals, cmap='viridis')
    # plt.colorbar()
    # plt.show()
    
    erg_ctrl    = RTErgodicControl(model, horizon=15, num_basis=5, batch_size=-1)
    
    # setting the phik on the ergodic controller
    erg_ctrl.phik = convert_phi2phik(erg_ctrl.basis, grid_vals, grid)

    print('--- simulating ergodic coverage ---')
    log = {'trajectory' : []}
    tf = 200                                                                                                                                                                                       
    state = env.reset()

    plt.figure(1)
    # xy, vals = t_dist.get_grid_spec()
    plt.title('ergodic coverage')
    plt.imshow(grid_vals, cmap='viridis')
    
    Erg_list = []
    for t in range(tf):
        ctrl = erg_ctrl(state)
        print(np.linalg.norm(ctrl))
        Erg_list.append(erg_ctrl.Erg_metric)
        state = env.step(ctrl)
        log['trajectory'].append(state)
        
        # plt.scatter(50*state[0], 50*state[1])
         # 暂停绘图并刷新窗口
        # plt.pause(0.01) 

    print('--- finished simulating ---')
    xt = np.stack(log['trajectory'])
    plt.scatter(50*xt[:tf,0], 50*xt[:tf,1])
    # plt.pause(3) 
    
    # trajectory statistics
    path = xt[:tf,model.explr_idx]
    ck = convert_traj2ck(erg_ctrl.basis, path) # coeffieient 25
    val = convert_ck2dist(erg_ctrl.basis, ck, grid) 
    
    plt.title('time averaged statistics')
    plt.imshow(val.reshape(50,50), cmap='viridis')
    plt.colorbar()
    plt.show()
    
    plt.figure(2)
    x = range(len(Erg_list))
    plt.plot(x, Erg_list, marker='o', linestyle='-', color='b')

    plt.title('Erg_metric')
    plt.xlabel('time')
    plt.ylabel('metric value')
    plt.show()

    plt.figure(3)
    plt.title('Fourier reconstruction of target distribution')
    phi_conv = convert_phik2phi(erg_ctrl.basis, erg_ctrl.phik, grid)
    plt.imshow(phi_conv.reshape(50,50), cmap='viridis')

    plt.show()