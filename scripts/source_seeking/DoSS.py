from matplotlib.pyplot import pause
import numpy as np
import sklearn.gaussian_process.kernels as kernels
from tqdm import tqdm
from matplotlib import cm
from scipy.optimize import linear_sum_assignment
from source_seeking_module.bayesian_optimization import BayesianOptimizationCentralized
from source_seeking_module.benchmark_functions_2D import *
from environment.environment_and_measurement import Environment
from combined_robot_GMES import Robot
from typing import List
from itertools import permutations


algorithm='doss'
Update_target_per_iter = False

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

# Bayesian optimization object
environment = Environment( 1 )

n_workers = 3
robot_locations = np.array([[1,2], [2,2], [3, 1]], dtype=float)
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

# start experiments!
query = None
found_source = []
end = False

reach_target = True
targets = None
# for iteration in tqdm(range(101), position = 1, leave = None):
for iteration in range(500):
    print("Iteration: ", iteration)
    ## Step 1: Train
    BO.train_gp_query(query_x=query, query_y=environment.sampling([query]))
    
    beta = 3 - 0.0019 * iteration
    if Update_target_per_iter == False:
        ## Step 2: Get targets
        if reach_target:
            # print(beta)
            targets = BO.find_next_query(iteration, beta=beta)
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
        print("change targets!")        
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
    print(found_source_set)
    if found_source_set == environment.SOURCE_SET:
        end = True
        
        
    # Plot optimization step
    if iteration % 20 == 0 or end:
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

        # trajectory
        # targets
        targets = np.array(targets)
        x_coords = targets[:, 0] 
        y_coords = targets[:, 1] 
        for i in range(BO.n_workers):
            color = COLORS[i % len(COLORS)]
            trajectory = Robots[i].trajectory.copy()
            trajectory.append(trajectory.pop(0))
            trajectory = np.array(trajectory)

            axs[0].plot(trajectory[:, 0], trajectory[:, 1], color=color, zorder=1)  
            axs[0].scatter(trajectory[:, 0], trajectory[:, 1], c=color, zorder=2)

            axs[1].scatter(x_coords[targets_indices[i]], y_coords[targets_indices[i]], s=80, c=color, marker='x', zorder=3)
        

        axs[0].set_xlabel('X Label')
        axs[0].set_ylabel('Y Label')
        axs[0].set_title('Trajectory and Ground Truth')
        
        cp2 = axs[1].contourf(X_plot, Y_plot, ucb.reshape(X_plot.shape),  cmap = cm.coolwarm)
        cbar = plt.colorbar(cp2, ax=axs[1])  # 添加颜色条


        axs[1].set_xlabel('X Label')
        axs[1].set_ylabel('Y Label')
        axs[1].set_title('Mean value')

        plt.show()
        if end:
            print("END NOW")
            pause(5)
            break


