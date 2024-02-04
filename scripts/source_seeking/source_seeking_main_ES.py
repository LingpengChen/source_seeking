import numpy as np
np.random.seed(10)

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from utils import find_peak, calculate_wrmse
from vonoroi_utils import voronoi_neighbours
from scipy.spatial import Voronoi, voronoi_plot_2d

from robot import Robot

## Initilize environment
from environment_and_measurement import Environment   
from IPython.display import clear_output

import argparse

COLORS = ['blue', 'green', 'red', 'purple', 'orange', 'black']  

DEBUG = False

def unique_list(redundant_list):
    tuple_list = [tuple(list(item)) for item in redundant_list]
    unique_list = [list(item) for item in set(tuple_list)]
    return unique_list

def experiment():
        
    #############################################################
    # Record the experiment
    rmse_values = []
    end = False
    WRMSE = True
    
    #############################################################
    ## start source seeking!
    for iteration in range(1000):
        ## 1 update vonoroi tessellation
        if True:
            # 1-1 find the neighbours of each robot
            for i in range(robo_num):
                robot_locations[i] = Robots[i].get_trajectory()[-1]
            neighbour_list = voronoi_neighbours(robot_locations)
            # 1-2 update voronoi cell info to each agents and exchange samples 
            exchange_dictionary_X = {}
            exchange_dictionary_y = {}
            
            for i in range(robo_num):
                temp_dictionary_X, temp_dictionary_y = Robots[i].voronoi_update(neighbour_list[i], robot_locations) 
                for k, v in temp_dictionary_X.items():
                    if k not in exchange_dictionary_X:
                        exchange_dictionary_X[k] = []
                    exchange_dictionary_X[k].extend(v)
                    
                for k, v in temp_dictionary_y.items():
                    if k not in exchange_dictionary_y:
                        exchange_dictionary_y[k] = []
                    exchange_dictionary_y[k].extend(v)
            # 1-3 receive samples
            for i in range(robo_num):
                if i in exchange_dictionary_X:
                    Robots[i].receive_samples(exchange_dictionary_X[i], exchange_dictionary_y[i])
        
        ## 2. train and estimate!
        μ_estimation = np.zeros(test_resolution)
        ucb = np.zeros(test_resolution)
        peaks = []
        LCB_list = []
        for i in range(robo_num):
            μ_partial, ucb_partial = Robots[i].gp_learn_and_get_acquisition(ucb_coeff=2)
            μ_estimation += μ_partial
            ucb += ucb_partial
            
            sources, lcb = Robots[i].estimate_source(lcb_coeff=2)
            peaks += sources
            LCB_list += lcb 
        
        ## 3. communication ck, phi_k and consensus
        for k in range(5):
            ck_pack = {}
            phik_pack = {}
            found_source = []
            for i in range(robo_num):
                found_source += Robots[i].send_out_source_cord()
                ck_pack[i] = Robots[i].send_out_ck()
                phik_pack[i] = Robots[i].send_out_phik()
            found_source = unique_list(found_source)


            for i in range(robo_num):
                Robots[i].receive_ck_consensus(ck_pack.copy()) 
                Robots[i].receive_source_cord(found_source)
                ucb_changed = Robots[i].receive_phik_consensus(phik_pack.copy()) 
        
        # determine whether all the source have been found
        found_source_set = {tuple(item) for item in found_source}
        if found_source_set == environment.SOURCE_SET:
            end = True
            
        ## 4. Move and taking samples!
        targets = []
        for i in range(robo_num):
            if iteration < 5:
                setpts = Robots[i].get_nextpts(control_mode = "ES_UNIFORM")
            else:
                setpts = Robots[i].get_nextpts(control_mode = "NORMAL")
                target = Robots[i].target
                if target:
                    targets.append(target)
        
        if DEBUG:
            print(iteration)
            print(found_source_set)
            print("=====================")    
        
    ####################################################################
        
        if WRMSE:
            rmse = calculate_wrmse(μ_estimation, environment.get_gt(X_test).reshape(test_resolution))
            rmse_values.append(rmse)
            
        else:
            rmse = np.sqrt(np.mean((μ_estimation - environment.get_gt(X_test).reshape(test_resolution)) ** 2))
            rmse_values.append(rmse)
        
    ## Visualize
        # if iteration == 40 or (iteration >= 45 and iteration % 5 == 0 and SHOWN) or end:
        if DEBUG or end:
            sizes = 5  # 可以是一个数字或者一个长度为N的数组，表示每个点的大小              
            # # 设置图表的总大小
            fig, axs = plt.subplots(1, 5, figsize=(24, 10), subplot_kw={'projection': '3d'})
            axs[0].remove()  # 移除第一个子图的3D投影
            axs[0] = fig.add_subplot(1, 5, 1)  # 添加一个2D子图

            ## Start plotting
            
            ## 在第1个子图上绘制2D图
            if True:   
                contour = axs[0].contourf(X_test_xx, X_test_yy, environment.get_gt(X_test).reshape(test_resolution))

                # 1-1) plot Voronoi tessellation
                vor = Voronoi(robot_locations)
                voronoi_plot_2d(vor, ax=axs[0], show_vertices=False)
                axs[0].set_xlim([0, 10])  # 你可以根据需要调整这些值
                axs[0].set_ylim([0, 10])  
                
                # 1-2) trajectory
                for i in range(robo_num):
                    color = COLORS[i % len(COLORS)]
                    trajectory = np.array(Robots[i].trajectory)
                    axs[0].plot(trajectory[:, 0], trajectory[:, 1], color=color, zorder=1)  # 设置线条颜色
                    axs[0].scatter(trajectory[:, 0], trajectory[:, 1],s=sizes, c=color, zorder=2)

                # 1-3) sources ground truth
                x_coords = environment.SOURCES[:, 0]
                y_coords = environment.SOURCES[:, 1]
                axs[0].scatter(x_coords, y_coords, s=10, c='black', marker='x', zorder=3)
                for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                    axs[0].text(x, y, f"{environment.SOURCE_VALUE[i]:.2f}", fontsize=6, ha='right', va='bottom')
                
                # 1-4) estimated sources
                if (len(peaks)!=0):
                    peaks = np.array(peaks)
                    x_coords = peaks[:, 0] 
                    y_coords = peaks[:, 1] 
                    axs[0].scatter(x_coords, y_coords, s=10, c='red', marker='x', zorder=3)
                    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                        axs[0].text(x, y, f"{LCB_list[i]:.2f}", c='red', fontsize=6, ha='right', va='top')

                if (len(targets)!=0):
                    targets = np.array(targets)
                    x_coords = targets[:, 0] 
                    y_coords = targets[:, 1] 
                    axs[0].scatter(x_coords, y_coords, s=10, c='green', marker='x', zorder=3)
                
                # axs[0].legend(loc="lower left")
                axs[0].set_aspect('equal')


            # 第2个子图 ground truth
            surf2 = axs[1].plot_surface(X_test_xx, X_test_yy, environment.get_gt(X_test).reshape(test_resolution), cmap='viridis', edgecolor='k', linewidth=0.5)
            zmin = 0  # 设置 z 轴的最小值
            zmax = 0.3  # 设置 z 轴的最大值
            axs[1].set_zlim([zmin, zmax])


            # 第3个子图 estimated distribution
            surf3 = axs[2].plot_surface(X_test_xx, X_test_yy, μ_estimation, cmap='viridis', edgecolor='k', linewidth=0.5)
            zmin = 0  # 设置 z 轴的最小值
            zmax = 0.3  # 设置 z 轴的最大值
            axs[2].set_zlim([zmin, zmax])
  
            # 第4个子图
            
            surf4 = axs[3].plot_surface(X_test_xx, X_test_yy, ucb, cmap='viridis', edgecolor='k', linewidth=0.5)
            fig.colorbar(surf4, ax=axs[3], pad=0.2, shrink=0.4)
            
            surf5 = axs[4].plot_surface(X_test_xx, X_test_yy, ucb_changed, cmap='viridis', edgecolor='k', linewidth=0.5)
            fig.colorbar(surf5, ax=axs[4], pad=0.2, shrink=0.4)
            # zmin = 0  # 设置 z 轴的最小值
            # zmax = 0.3  # 设置 z 轴的最大值
            # axs[3].set_zlim([zmin, zmax])

            # 设置标签和标题 
            axs[0].set_xlabel('X Label')
            axs[0].set_ylabel('Y Label')
            axs[0].set_title('Trajectory')
            
            axs[1].set_xlabel('X Label')
            axs[1].set_ylabel('Y Label')
            axs[1].set_zlabel('Z Label')
            axs[1].set_title('Ground Truth')

            axs[2].set_xlabel('X Label')
            axs[2].set_ylabel('Y Label')
            axs[2].set_zlabel('Z Label')
            axs[2].set_title('Mean value')
            
            axs[3].set_xlabel('X Label')
            axs[3].set_ylabel('Y Label')
            axs[3].set_zlabel('Z Label')
            axs[3].set_title('UCB')
            
            axs[4].set_xlabel('X Label')
            axs[4].set_ylabel('Y Label')
            axs[4].set_zlabel('Z Label')
            axs[4].set_title('UCB_changed')
            # plt.show()
            # if iteration == 2:
            if end:
                plt.savefig(save_img_path)
                plt.close()
                with open(save_rmse_path, 'w', encoding='utf-8') as file:
                    for item in rmse_values:
                        file.write(str(item) + '\n')
                break

    print("Experiment_", experiment_case, "Finished!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source_index', type=int, help='choose the sources topology you want')
    # parser.add_argument('source_index', type=int, help='choose the sources topology you want',  nargs='?', default=1)
    args = parser.parse_args()
    experiment_case = args.source_index

    environment = Environment(experiment_case)

    save_img_path = "/home/clp/catkin_ws/src/source_seeking/record/experiment_case_" + str(experiment_case) + ".png"
    save_rmse_path = "/home/clp/catkin_ws/src/source_seeking/record/experiment_case_" + str(experiment_case) + ".txt"

    #############################################################
    ## Define the source field
    
    FIELD_SIZE_X = 10
    FIELD_SIZE_Y = 10
    x_min = (0, 0)
    x_max = (FIELD_SIZE_X, FIELD_SIZE_Y)
    # 用于采样的参数
    if True:
        # Predict points at uniform spacing to capture function
        test_resolution = [50, 50]
        X_test_x = np.linspace(x_min[0], x_max[0], test_resolution[0])
        X_test_y = np.linspace(x_min[1], x_max[1], test_resolution[1])
        X_test_xx, X_test_yy = np.meshgrid(X_test_x, X_test_y)
        X_test = np.vstack(np.dstack((X_test_xx, X_test_yy)))
    
    #############################################################
    ## Initial robots
    # Give Robots Prior knowledge (all the same)
    if True:
        n_train = 0
        # Number of training points and testing points
        X_train=None
        y_train=None
        if (n_train):
            # Sample noisy observations (X1, y1) from the function for each of the GP components
            X_train = np.random.uniform(x_min, x_max, size=(n_train, 2))
            y_train = environment.sampling(X_train)
    # Set 
    if True:
        robo_num = 3
        # robot_locations = [[3,3], [5, 4], [7,3]]
        robot_locations = [[1,2], [2, 2], [3,1]]
        Robots = []
        for index in range(robo_num):
            # initialize the thread
            instance = Robot(robot_locations[index], index, environment, test_resolution=[50,50]) # resolution is 50 * 50
            instance.receive_prior_knowledge(X_train, y_train)
            Robots.append(instance)

    experiment()
    
   