import numpy as np

def distance(point1, point2):
    # 将两个点转换为NumPy数组以确保正确的数学操作
    point1 = np.array(point1)
    point2 = np.array(point2)
    # 计算两点间的差值
    delta = point1 - point2
    # 计算欧几里得距离
    dist = np.sqrt(np.dot(delta, delta))
    return dist

# 示例用法
p1 = [1, 2]
p2 = [4, 5]
print("Distance between p1 and p2:", distance(p1, p2))
