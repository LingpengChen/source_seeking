from matplotlib.pyplot import pause
import numpy as np
import sklearn.gaussian_process.kernels as kernels
from tqdm import tqdm
from matplotlib import cm
from scipy.optimize import linear_sum_assignment
from source_seeking_module.bayesian_optimization import BayesianOptimizationCentralized
from source_seeking_module.benchmark_functions_2D import *
from environment.environment_and_measurement_7 import Environment, ROBOT_INIT_LOCATIONS_case
from combined_robot_GMES import Robot
from typing import List
from itertools import permutations
from utils.analysis_utils import calculate_wrmse
import argparse
import sys


algorithm='es'
Update_target_per_iter = False
Record = True

def assign_targets_indices(robot_locations, targets):

    # 计算机器人和目标之间的欧几里得距离
    def calculate_distances(targets, robot_locations):
        return np.linalg.norm(targets - robot_locations, axis=1)

    # 遍历所有可能的分配方案
    min_max_distance = float('inf')
    best_assignment = None
    for permutation in permutations(range(len(targets))):
        distances = calculate_distances(targets[list(permutation)], robot_locations)
        max_distance = np.max(distances)
        if max_distance < min_max_distance:
            min_max_distance = max_distance
            best_assignment = permutation

    return best_assignment

def unique_list(redundant_list):
    tuple_list = [tuple(list(item)) for item in redundant_list]
    unique_list = [list(item) for item in set(tuple_list)]
    return unique_list


COLORS = ['blue', 'green', 'red', 'purple', 'orange', 'black']  # 你可以添加更多颜色
# Set seed
np.random.seed(0)


def main():
    # start experiments!
    query = None
    found_source = []
    end = False
    # record
    wrmse_values = []
    found_src_num = 0
    src_found_iteration =[]

    reach_target = True
    targets = None
    # for iteration in tqdm(range(101), position = 1, leave = None):
    for iteration in range(500):
        if not Record:
            print("Iteration: ", iteration)
        else:
            print_progress_bar(experiment_case, iteration)
        ## Step 1: Train
        BO.train_gp_query(query_x=query, query_y=environment.sampling([query]))
        
        # beta = 1 - 0.004 * iteration
        # if beta < 0.4: beta = 0.4
        beta = 3 - 0.02 * iteration
        if beta < 0.4: beta = 0.4
        # beta = 3 - 0.0019 * iteration
        if Update_target_per_iter == False:
            ## Step 2: Get targets
            if reach_target:
                # print(beta)
                queries = BO.find_next_query(iteration, beta=beta)
                targets = queries
                # Use Hungarian algorithm to assign nearest targets for each agent
                targets_indices = assign_targets_indices(robot_locations, targets)
                # now we have new target
                reach_target = False
            
            
            ## Step 3: Move to targets
            flag_all_reach = True
            for index in range(robo_num):
                # initialize the thread
                setpts, visited_peaks_cord = Robots[index].get_nextpts_from_target(targets[targets_indices[index]])
                # Check whether reach target
                dis2target = np.linalg.norm(np.array(targets[targets_indices[index]]) - np.array(robot_locations[index]))
                if (dis2target > 0.5):
                    flag_all_reach = False
                # Move robot
                robot_locations[index] = setpts[0]
                found_source += visited_peaks_cord
            
            if flag_all_reach:
                reach_target = True
        
        else:
            ## Step 2: Get targets
            targets = BO.find_next_query(iteration, beta=beta)
            # Use Hungarian algorithm to assign nearest targets for each agent
            targets_indices = assign_targets_indices(robot_locations, targets)
            
            ## Step 3: Move to targets
            for index in range(robo_num):
                # initialize the thread
                setpts, visited_peaks_cord = Robots[index].get_nextpts_from_target(targets[targets_indices[index]])
                # Move robot
                robot_locations[index] = setpts[0]
                found_source += visited_peaks_cord
            
        
        
        query = robot_locations.copy()    
        # Determine whether have found all sources
        found_source = unique_list(found_source)
        found_source_set = {tuple(item) for item in found_source}
        if (len(found_source_set) > found_src_num):
            found_src_num = len(found_source_set)
            src_found_iteration.append(iteration)
            if not Record:
                print(found_source_set)
            
        if found_source_set == environment.SOURCE_SET:
            end = True
        
        if Record:
            mu, std = BO.model.predict(BO._grid, return_std=True)
            rmse = calculate_wrmse(mu, environment.get_gt(BO._grid))
            wrmse_values.append(rmse)
            
        # Plot optimization step
        if (iteration % 100 == 0 and not Record) or end:
            fig, axs = plt.subplots(1, 2, figsize=(20, 10))
            first_param_grid = np.linspace(0, 10, 100)
            second_param_grid = np.linspace(0, 10, 100)
            X_plot, Y_plot = np.meshgrid(first_param_grid, second_param_grid, indexing='ij')
            x = np.array(BO.X)
            y = np.array(BO.Y)
            mu, std = BO.model.predict(BO._grid, return_std=True)
            
            
            ucb = mu + beta * std
            # Objective plot
            N = 100
            Y_obj = [environment.get_gt(i) for i in BO._grid]
            clev1 = np.linspace(min(Y_obj), max(Y_obj),N)
            cp1 = axs[0].contourf(X_plot, Y_plot, np.array(Y_obj).reshape(X_plot.shape), clev1,  cmap = cm.coolwarm)

            targets = np.array(targets)
            found_sources = np.array(list(found_source_set))
            x_coords = targets[:, 0] 
            y_coords = targets[:, 1] 
            for i in range(BO.n_workers):
                color = COLORS[i % len(COLORS)]
                trajectory = Robots[i].trajectory.copy()
                # trajectory.append(trajectory.pop(0))
                trajectory = np.array(trajectory)

                axs[0].plot(trajectory[:, 0], trajectory[:, 1], color=color, zorder=1)  
                axs[0].scatter(trajectory[:, 0], trajectory[:, 1], c=color, zorder=2)
                if len(found_sources)>0:
                    axs[0].scatter(found_sources[:, 0], found_sources[:, 1],s=80,  c='black',  marker='x', zorder=3)

                axs[1].scatter(x_coords[targets_indices[i]], y_coords[targets_indices[i]], s=80, c=color, marker='x', zorder=3)
            

            axs[0].set_xlabel('X Label')
            axs[0].set_ylabel('Y Label')
            axs[0].set_title('Trajectory and Ground Truth')
            
            cp2 = axs[1].contourf(X_plot, Y_plot, ucb.reshape(X_plot.shape),  cmap = cm.coolwarm)
            # cp2 = axs[1].contourf(X_plot, Y_plot, mu.reshape(X_plot.shape),  cmap = cm.coolwarm)
            cbar = plt.colorbar(cp2, ax=axs[1])  # 添加颜色条


            axs[1].set_xlabel('X Label')
            axs[1].set_ylabel('Y Label')
            axs[1].set_title('Mean value')

            if end:
                if Record:
                    plt.savefig(save_img_path)
                    plt.close()
                    with open(save_rmse_path, 'w', encoding='utf-8') as file:
                        for item in wrmse_values:
                            file.write(str(item) + '\n')
                    with open(save_found_src_path, 'w', encoding='utf-8') as file:
                        for item in src_found_iteration:
                            file.write(str(item) + '\n')
                else:
                    plt.show()
                    print("END NOW")
                break

def print_progress_bar(index, iteration):
    position = index%4+1
    sys.stdout.write(f'\033[{position}B\r')  # Move to correct position
    sys.stdout.write(f'Case {index}: {iteration}\033[{position}A\r')  # Print progress bar and move back up
    sys.stdout.flush()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('env_index', type=int, help='choose the sources topology you want', nargs='?', default=1)
    parser.add_argument('robot_ini_loc_index', type=int, help='choose the robot initial location',  nargs='?', default=1)
    args = parser.parse_args()
    env_index = int(args.env_index)
    robot_ini_loc_index = int(args.robot_ini_loc_index)
    
    experiment_case = len(ROBOT_INIT_LOCATIONS_case)*env_index+robot_ini_loc_index
    print("Start experiment case_", experiment_case)
    
    environment = Environment( env_index )
    
    save_rmse_path = "/home/clp/catkin_ws/src/source_seeking/record/7sources/GMES/rmse" + str(experiment_case) + ".txt"
    save_found_src_path = "/home/clp/catkin_ws/src/source_seeking/record/7sources/GMES/src" + str(experiment_case) + ".txt"
    save_img_path = "/home/clp/catkin_ws/src/source_seeking/record/7sources/GMES/img" + str(experiment_case) + ".png"

    n_workers = 3
    # robot_locations = np.array([[1,2], [2,2], [3, 1]], dtype=float)
    robot_locations = ROBOT_INIT_LOCATIONS_case[robot_ini_loc_index]
    n_pre_samples = 0
    # robot_locations = np.array([[1,2]])
    BO = BayesianOptimizationCentralized(domain=np.array([[0, 10], [0, 10]]),
                                n_workers=n_workers,
                                acquisition_function=algorithm,
                                env = environment
                                )
    BO.initialize(robot_locations, environment.sampling(robot_locations), n_pre_samples=n_pre_samples)

    robo_num = 3
    Robots: List[Robot] = []
    for index in range(robo_num):
        # initialize the thread
        instance = Robot(robot_locations[index], index, environment, test_resolution=[50,50]) # resolution is 50 * 50
        Robots.append(instance)
        
    main()