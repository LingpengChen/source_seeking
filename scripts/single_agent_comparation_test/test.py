import numpy as np

matrix = np.array([[1,1,1],[1,1,1],[1,1,0]])
# 假设 matrix 是一个 50x50 的 NumPy 矩阵

# 您的索引列表
retrieve_list = [[1, 1], [2, 2]]

# 将列表转换为元组，用于索引
# 您需要将行和列索引分开，以便 NumPy 能够正确理解它们


# 使用高级索引从矩阵中检索元素
retrieved_elements = matrix[[i[0] for i in retrieve_list], [i[1] for i in retrieve_list]]

print(retrieved_elements)
