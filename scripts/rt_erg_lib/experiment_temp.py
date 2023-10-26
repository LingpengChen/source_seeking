from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import numpy as np

# 创建点集
points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# 计算Voronoi tessellation
vor = Voronoi(points)

# 绘制结果
voronoi_plot_2d(vor)
plt.show()
