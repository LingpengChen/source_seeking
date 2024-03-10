import numpy as np
from itertools import permutations

# 假设有三个目标和三个机器人的坐标
targets = np.array([[0, 2], [1, 2], [2, 2]])
robot_locations = np.array([[1, 0], [0, 0],  [2, 0]])

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
# 输出结果
best_assignment = assign_targets_indices(robot_locations, targets)
print(f"最佳分配方案：{best_assignment}")
