import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist

def get_robot_from_region(dictionary, target_value):
    keys_with_value = [key for key, value in dictionary.items() if value == target_value]
    return keys_with_value[0]

def voronoi_neighbours(robots):
    
    if len(robots) > 1:
        neighbour_list = [] # neighbours for each robot
        
        # 创建 Voronoi 对象
        vor = Voronoi(robots)

        # 绘制 Voronoi 图
        # voronoi_plot_2d(vor, show_vertices=True)
        
        robot_region_dict = {}
        # 在每个输入点旁边显示其对应的区域索引
        for i, (x, y) in enumerate(robots):
            # plt.text(x, y, f'robot {i}', color='red', fontsize=12, ha='right')
            robot_region_dict[i] = vor.point_region[i]
            

        # 找到并打印每个机器人的邻居区域
        for i in range(len(robots)):
        
            neighbors = []
            for ridge in vor.ridge_points:
                if i in ridge:
                    if ridge[0] == i:
                        neighbors.append(vor.point_region[ridge[1]])
                    else:
                        neighbors.append(vor.point_region[ridge[0]])
            neighbors = [get_robot_from_region(robot_region_dict, region) for region in neighbors]
            # print(f'Robot {i} in Region {vor.point_region[i]} has neighbors: {[get_robot_from_region(robot_region_dict, region) for region in neighbors]}')
            neighbors.sort()
            neighbour_list.append(neighbors)
    else:
        neighbour_list = [[]]       
    return neighbour_list

def generate_voronoi(robots, index):
    grid_size = 20
    voronoi = np.zeros((grid_size, grid_size))

    # 生成一个20x20的格点矩阵
    grid_points = np.array([(x, y) for x in range(grid_size) for y in range(grid_size)])

    # 计算每个格点到所有机器人的距离
    distances = cdist(grid_points, robots, metric='euclidean')

    # 找到每个格点最近的机器人
    closest_robot = np.argmin(distances, axis=1)

    print(closest_robot)
    
    # 标记距离 [0,0] 机器人最近的格点为 1
    for i, robot_index in enumerate(closest_robot):
        if robot_index == index:
            y, x = grid_points[i]
            voronoi[x, y] = 1

    return voronoi

# 设置标题和显示图像
if __name__ == '__main__':
    # robots = [[0, 0], [1, 3], [3, 3], [3, 1]]
    robots = np.array([[0, 0], [0, 20], [20, 20]])
    robots = robots / 10
    print(robots)
    neighbour_list = voronoi_update(robots, 1)
    print(neighbour_list)
    
    index_mat0 = generate_voronoi(robots, 0)
    index_mat1 = generate_voronoi(robots, 1)
    index_mat2 = generate_voronoi(robots, 2)
    plt.subplot(1, 3, 1)

    plt.imshow(index_mat0, origin='lower', cmap='viridis')
    plt.title('Robot 0 Voronoi Cell')

    plt.subplot(1, 3, 2)
    plt.imshow(index_mat1, origin='lower', cmap='viridis')
    plt.title('Robot 1 Voronoi Cell')

    plt.subplot(1, 3, 3)
    plt.imshow(index_mat2, origin='lower', cmap='viridis')
    plt.title('Robot 2 Voronoi Cell')
    print(index_mat0+index_mat1+index_mat2)

    # 显示所有 subplot
    plt.tight_layout()  # 自动调整 subplot 参数，以给定指定的填充
    plt.show()
    