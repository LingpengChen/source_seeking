from matplotlib import pyplot as plt
import numpy as np
from numpy.random.mtrand import beta
from environment import Environment
import itertools
COLORS = ['blue', 'green', 'red', 'purple', 'orange', 'black']  

def find_index_of_value(nums, value):
    return [i for i, num in enumerate(nums) if num == value]


def assign_targets(robots_locations, possible_target_list):
    def distance(p1, p2):
        return np.linalg.norm(p1 - p2)
    
    def find_nearest_targets(robot_cord, possible_target_list, n=3):
        distances = np.linalg.norm(possible_target_list - robot_cord, axis=1)
        nearest_indices = np.argsort(distances)[:n]
        return possible_target_list[nearest_indices]

    min_distance = float('inf')
    best_assignment = None

    target_list = np.empty((0, 2))  # 假设每个目标都是二维的
    for robot_cord in robots_locations: # find the shortest location 
        temp = find_nearest_targets(robot_cord, possible_target_list)  # 找到最近的目标
        target_list = np.vstack((target_list, temp))  # 将新的目标添加到列表中

    _, unique_indices = np.unique(target_list, axis=0, return_index=True)
    target_list = target_list[unique_indices]
           
    for targets in itertools.permutations(target_list, len(robots_locations)): # all possible permutation of targets with same mu
        total_distance = sum(distance(robot, target) for robot, target in zip(robots_locations, targets))
        if total_distance < min_distance:
            min_distance = total_distance
            best_assignment = targets

    assignment = []
    for i in best_assignment:
        assignment.append(i.tolist())
    return np.array(assignment)


class Doss():
    def __init__(self, map_resolution):
        map_size = map_resolution**2
        self.V = np.diag([0.01] * 3)
        # Initial estimate of the state for the entire field
        self.phi_k = np.zeros((map_size, 1))
        self.mu_k = None
        # Initial covariance matrix (assuming initial uncertainty for the entire field)
        self.Sigma_k = np.eye(map_size)
        
        self.robots_location = np.array([[0.5,1.5],[1.5,1.5],[1.5,0.5]])
        self.trajectory = [[self.robots_location[0].tolist()], [self.robots_location[1].tolist()], [self.robots_location[2].tolist()]]
        
        
    def kalman_consensus_filter(self, z_k, H_k):
        Y_k = H_k.T @ np.linalg.inv(self.V) @ H_k
        self.Sigma_k = np.linalg.inv(np.linalg.inv(self.Sigma_k) + Y_k)      
        y_k = H_k.T @ np.linalg.inv(self.V) @ z_k
        y_k = y_k.reshape((-1, 1))
        self.phi_k = self.phi_k + self.Sigma_k @ (y_k - Y_k @ self.phi_k)
        return self.phi_k, self.Sigma_k

    def d_ucb(self, beta):
        #D-UCB
        self.mu_k = self.phi_k + beta * np.sqrt(np.diag(self.Sigma_k)).reshape((-1, 1))
        return self.mu_k
    
    def select_target(self):
        mu_k_list = self.mu_k.flatten()
        max_mu_values = np.sort(mu_k_list)[:-4:-1]
        
        if max_mu_values[0] == max_mu_values[1] and max_mu_values[1] == max_mu_values[2]:
            index_list = find_index_of_value(mu_k_list, max_mu_values[0]) # index of position has the highest mu
            # for index in index_list(index_list):
            #     if X[i]
            possible_target_list = np.array([X[i].tolist() for i in index_list])
            best_assignment = assign_targets(self.robots_location, possible_target_list)
        else:
            index_list = []
            for i in range(3):
                index_list.append( find_index_of_value(mu_k_list, max_mu_values[i])[0] )
            possible_target_list = np.array( [X[i] for i in index_list] )
            best_assignment = assign_targets(self.robots_location, possible_target_list)
        return best_assignment
            # self.robots_location
        # mu_values = np.sort(mu_k_list)
        # for mu_value in mu_values[:-4:-1]:
        #     index_list = find_index_of_value(mu_k_list ,mu_value)
        #     cord_list = [X[i] for i in index_list]
        #     print(cord_list)
        #     break
    
    def update_robot_location(self, best_assignment):
        self.robots_location = best_assignment
        for i, cord in enumerate( self.robots_location ):
            self.trajectory[i].append(cord.tolist())


if __name__ == '__main__':
    field_size = 10
    map_resolution = 50
    map_step_size = field_size/map_resolution
    X_test_x = np.linspace(map_step_size/2, field_size-map_step_size/2, map_resolution)
    X_test_y = X_test_x
    X_test_xx, X_test_yy = np.meshgrid(X_test_x, X_test_y)
    X = np.vstack(np.dstack((X_test_xx, X_test_yy)))

    env = Environment(field_size, map_resolution)
    Y = env.f(X)
    doss = Doss(map_resolution)
    for k in range(100):
        
        H_k = env.get_H(doss.robots_location)
        z_k = env.sampling(doss.robots_location)
        phi_k, Sigma_k = doss.kalman_consensus_filter(z_k, H_k)
        mu_k = doss.d_ucb(beta=0.2)
        targets = doss.select_target()

        plt.gca().set_aspect('equal', adjustable='box')
        plt.contourf(X_test_xx, X_test_yy, mu_k.reshape([map_resolution, map_resolution]), cmap='coolwarm', edgecolor='none', levels=100)
        for i in range(3):
            color = COLORS[i % len(COLORS)]
            tra = np.array(doss.trajectory[i])
            plt.scatter(tra[:, 0], tra[:, 1], c=color, marker='x')
            plt.plot(tra[:, 0], tra[:, 1], color=color, zorder=1)  
            
        plt.xlim(0, field_size)
        plt.ylim(0, field_size)
        plt.show()
 
        doss.update_robot_location(targets)
    # # # 设置图像的长宽比为一致
    