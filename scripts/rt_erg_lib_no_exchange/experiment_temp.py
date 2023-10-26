import numpy as np
from scipy.spatial import Voronoi, distance

# 定义 Voronoi 图的生成点
robot_location = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# 创建 Voronoi 图
vor = Voronoi(robot_location)

# 定义要检查的点
test_point = [[2,2],[2,0],[3,0.5]]

# 计算 test_point 到每个生成点的距离
exchange_dictionary = {}
self_index = 0
closest_point_index_list = []
for sample in test_point:
    distances = [distance.euclidean(sample, robo) for robo in robot_location]
    # 找到距离最近的生成点的索引
    closest_point_index = np.argmin(distances)
    closest_point_index_list.append(closest_point_index)
    if closest_point_index != self_index:
        if closest_point_index not in exchange_dictionary:
            exchange_dictionary[closest_point_index] = []  # 如果不存在，则初始化一个空列表

        exchange_dictionary[closest_point_index].append(sample)
    print(closest_point_index_list)
    print(robot_location)
    
print(exchange_dictionary)
# 现在，closest_point_index 即为 test_point 所在的 Voronoi 单元的索引
# 假设 dicts 是包含所有你想合并的字典的列表
dicts = [
    {1: [[1, 2], [3, 4]], 2: [[5, 6]]},
    {1: [[7, 8], [9, 10]], 2: [[11, 12]]},
    # 添加更多字典如果需要
]

# 初始化一个空字典来存储合并的结果
merged_dict = {}

# 遍历 dicts 中的每个字典
for d in dicts:
    for k, v in d.items():
        # 如果键还不存在于 merged_dict 中，则创建一个新列表
        if k not in merged_dict:
            merged_dict[k] = []
        # 将当前字典的值添加到合并字典中对应的键
        merged_dict[k].extend(v)

print(merged_dict)
