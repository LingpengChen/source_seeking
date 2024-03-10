import numpy as np

def find_nearest_targets(robot_cord, possible_target_list, n=20):
    # 计算机器人与所有目标之间的距离
    distances = np.linalg.norm(possible_target_list - robot_cord, axis=1)
    
    # 获取距离最近的n个目标的索引
    nearest_indices = np.argsort(distances)[:n]
    
    # 返回距离最近的n个目标
    return possible_target_list[nearest_indices]

# 示例
robot_cord = np.array([1, 2])
possible_target_list = np.array([[4, 5], [7, 8], [1, 3], [10, 11], [13, 14], [16, 17]])

nearest_targets = find_nearest_targets(robot_cord, possible_target_list)
print(nearest_targets)
