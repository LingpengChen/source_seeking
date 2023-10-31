import numpy as np
from scipy.spatial import Voronoi, distance
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# rows, cols = 50, 50
# responsible_region = np.zeros((rows, cols))
# # 将矩阵的右半部分设置为 1
# responsible_region[:, cols//2:] = 100


# def kernel_initial(
#             σf_initial=1.0,         # covariance amplitude
#             ell_initial=1.0,        # length scale
#             σn_initial=0.1          # noise level
#         ):
#             return σf_initial**2 * RBF(length_scale=ell_initial, length_scale_bounds=(0.5, 5)) + WhiteKernel(noise_level=σn_initial)

# test_resolution = [50,50]
# field_size = [10 ,10]
# grid_2_r_w = np.meshgrid(np.linspace(0, 1, int(test_resolution[0])), np.linspace(0, 1, int(test_resolution[1])))
# grid = np.c_[grid_2_r_w[0].ravel(), grid_2_r_w[1].ravel()] #(2500,2)

# X_test_x = np.linspace(0, field_size[0], test_resolution[0])
# X_test_y = np.linspace(0, field_size[1], test_resolution[1])
# X_test_xx, X_test_yy = np.meshgrid(X_test_x, X_test_y)
# X_test = np.vstack(np.dstack((X_test_xx, X_test_yy))) #(50,50)

# gp = GaussianProcessRegressor(
#             kernel=kernel_initial(),
#             n_restarts_optimizer=10
#         )


# samples_X =np.array(
# [[0.03948266, 5.12192263]
 
#  ] )

# samples_Y = np.array(
# [0.0693098])


# gp.fit(samples_X, samples_Y)
# μ_test, σ_test = gp.predict(X_test, return_std=True)

# estimation = μ_test.reshape(test_resolution)
# # plt.imshow(estimation.reshape(test_resolution), cmap='viridis',origin='lower')
# # plt.show()

# # 创建 x, y 坐标
# x = np.linspace(0, 49, 50)
# y = np.linspace(0, 49, 50)
# x, y = np.meshgrid(x, y)

# # 创建 3D 图形
# fig, ax = plt.subplots(1, 2,  subplot_kw={'projection': '3d'})
# ax[0] = fig.add_subplot(121, projection='3d')
# ax[0].plot_surface(x, y, estimation, cmap='viridis')
# ax[0].set_zlim(0, 0.1)

# # estimation = μ_test.reshape(test_resolution) * responsible_region
# # plt.imshow(estimation.reshape(test_resolution), cmap='viridis',origin='lower')
# # plt.show()
# samples_X =np.array(
# [[0.03948266, 5.12192263],
#  [0.03948266, 5.12192263],
#  [0.03948266, 5.12192263],
#  [0.03948266, 5.12192263],
#  [0.03948266, 5.12192263],
#  [0.03948266, 5.12192263]
#  ] )

# samples_Y = np.array(
# [0.0693098,0.0693098,0.0693098,0.0693098,0.0693098,0.0693098])



# gp.fit(samples_X, samples_Y)
# μ_test, σ_test = gp.predict(X_test, return_std=True)
# estimation = μ_test.reshape(test_resolution)

# ax[1] = fig.add_subplot(122, projection='3d')
# ax[1].plot_surface(x, y, estimation, cmap='viridis')
# ax[1].set_zlim(0, 0.1)

# plt.show()

clp = np.array([1,3])
what = (clp is None)
print(what)