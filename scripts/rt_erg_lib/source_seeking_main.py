import numpy as np
np.random.seed(10)

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# from rt_erg_lib.controller import Controller
# from rt_erg_lib.utils import find_peak
from controller import Controller
from utils import find_peak

from vonoroi_utils import voronoi_neighbours
## Initilize environment
from scipy.stats import multivariate_normal

from scipy.spatial import Voronoi, voronoi_plot_2d
import threading


import rospy

COLORS = ['blue', 'green', 'red', 'purple', 'orange', 'black']  # 你可以添加更多颜色


def main():
        
    ## Define the source field
    if True:
        FIELD_SIZE_X = 10
        FIELD_SIZE_Y = 10

        source = np.array([[1,6], [8,7],[8,2], [5,6], [3,2]])
        source1 = multivariate_normal(source[0], 0.8*np.eye(2))
        source2 = multivariate_normal(source[1], np.eye(2))
        source3 = multivariate_normal(source[2], np.eye(2))
        source4 = multivariate_normal(source[3], np.eye(2))
        source5 = multivariate_normal(source[4], np.eye(2))
        

        f = lambda x: source1.pdf(x) + 1.1*source2.pdf(x) + source3.pdf(x) + 0.9*source4.pdf(x) + source5.pdf(x)
        source_value = f(source)

        x_min = (0, 0)
        x_max = (0+FIELD_SIZE_X, 0+FIELD_SIZE_Y)

    ## Initialize GP (uni-GP)
    if True:
        # Specify kernel with initial hyperparameter estimates
        def kernel_initial(
            σf_initial=1.0,         # covariance amplitude
            ell_initial=1.0,        # length scale
            σn_initial=0.1          # noise level
        ):
            return σf_initial**2 * RBF(length_scale=ell_initial) + WhiteKernel(noise_level=σn_initial)

        gp = GaussianProcessRegressor(
            kernel=kernel_initial(),
            n_restarts_optimizer=10
        )

    ## Define measurement 
    if True:
        # Define measurement noise model
        σ_noise = 0.001
        def measure(X_trains):
            return f(X_trains) + np.random.normal(0, σ_noise, size=(X_trains.shape[0],))

    ## 用于采样的参数
    if True:
        # Predict points at uniform spacing to capture function
        
        test_resolution = (50, 50)
        X_test_x = np.linspace(x_min[0], x_max[0], test_resolution[0])
        X_test_y = np.linspace(x_min[1], x_max[1], test_resolution[1])
        X_test_xx, X_test_yy = np.meshgrid(X_test_x, X_test_y)
        X_test = np.vstack(np.dstack((X_test_xx, X_test_yy)))

    ## Initial robots
    if True:
        robo_num = 3
        robot_locations = [[3,3], [5, 4], [7,3]]
        Robots = []
        trajectory = []
        for index in range(robo_num):
            # initialize the thread
            instance = Controller(start_position=robot_locations[index], index=index, resolution=[50,50], field_size=[10,10]) # resolution is 50 * 50
            Robots.append(instance)
            trajectory.append(np.empty((0,2)))

    ## Give Robots Prior knowledge (all the same)
    if True:
        # Number of training points and testing points
        n_train = 0

        if (n_train):
            # Sample noisy observations (X1, y1) from the function for each of the GP components
            X_train = np.random.uniform(x_min, x_max, size=(n_train, 2))
        else:
            X_train = np.array(robot_locations)

        y_train = f(X_train) + np.random.normal(0, σ_noise, size=(1,))
        print(y_train)
        gp.fit(X_train, y_train)

    rospy.init_node('main_function', anonymous=True)

    
    for iteration in range(100):
        
        if rospy.is_shutdown():
            break
        
        print(iteration)
        
        # 0-1 vonoroi tessellation
        # get neighbours
        robot_locations.clear()
        for i in range(robo_num):
            robot_locations.append(Robots[i].get_location())
        neighbour_list = voronoi_neighbours(robot_locations)
        # update voronoi cell info in each agents 
        for i in range(robo_num):
            Robots[i].voronoi_update(neighbour_list[i], robot_locations)
        
        # 0-2 communication and consensus
        ck_pack = {}
        for i in range(robo_num):
            ck_pack[i] = Robots[i].send_out_ck()

        for i in range(robo_num):
            Robots[i].receive_ck_consensus(ck_pack.copy()) 
            
        # 1. doing estimation based on the gp model trained in the last round
        μ_test, σ_test = gp.predict(X_test, return_std=True)
        μ_test_2D = μ_test.reshape(test_resolution)
        σ_test_2D = σ_test.reshape(test_resolution)
        peaks = find_peak(μ_test_2D)
        
        LCB_list = []
        for peak in peaks:
            LCB = μ_test_2D[peak[0]][peak[1]] - 2*σ_test_2D[peak[0]][peak[1]]
            LCB_list.append(LCB)
        
        # 2. UCB!
        phi_vals = μ_test + 0.5*σ_test
        phi_vals_2D = phi_vals.reshape(test_resolution)
        
        # 3. Move!
        for i in range(robo_num):
            setpts = Robots[i].get_nextpts(phi_vals)
            trajectory[i] = np.concatenate((trajectory[i], setpts), axis=0) 
        
            # 4. Take samples and add to dataset
            measurements = measure(setpts) 
            X_train = np.concatenate((X_train, setpts), axis=0)
            y_train = np.concatenate((y_train, measurements), axis=0)
            
        # 4. train!
        gp.fit(X_train, y_train)

# Visualize
        SHOWN = True
        if (iteration >= 0 and iteration % 10 == 0 and SHOWN):
            sizes = 5  # 可以是一个数字或者一个长度为N的数组，表示每个点的大小              
            # # 设置图表的总大小

            fig, axs = plt.subplots(1, 4, figsize=(24, 10), subplot_kw={'projection': '3d'})
            axs[0].remove()  # 移除第一个子图的3D投影
            axs[0] = fig.add_subplot(1, 4, 1)  # 添加一个2D子图

            ## Start plotting
            
            # 在第一个子图上绘制2D图
            contour = axs[0].contourf(X_test_xx, X_test_yy, f(X_test).reshape(test_resolution))

            # plot Voronoi tessellation
            vor = Voronoi(robot_locations)
            voronoi_plot_2d(vor, ax=axs[0], show_vertices=False)
            axs[0].set_xlim([0, 10])  # 你可以根据需要调整这些值
            axs[0].set_ylim([0, 10])  
            
            # trajectory
            for i in range(robo_num):
                color = COLORS[i % len(COLORS)]
                axs[0].plot(trajectory[i][:, 0], trajectory[i][:, 1], color=color, zorder=1)  # 设置线条颜色
                axs[0].scatter(trajectory[i][:, 0], trajectory[i][:, 1],s=sizes, c=color, zorder=2)
            
            # sources ground truth
            if (True):
                x_coords = source[:, 0]
                y_coords = source[:, 1]
                axs[0].scatter(x_coords, y_coords, s=10, c='black', marker='x', zorder=3)

                for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                    axs[0].text(x, y, f"{source_value[i]:.2f}", fontsize=6, ha='right', va='bottom')
            
            # estimated sources
            if (len(peaks)!=0):
                peaks = np.array(peaks)

                x_coords = FIELD_SIZE_X * peaks[:, 1] / test_resolution[0]
                y_coords = FIELD_SIZE_Y * peaks[:, 0] / test_resolution[1]

                axs[0].scatter(x_coords, y_coords, s=10, c='red', marker='x', zorder=3)

                # 在每个点旁边添加文本
                for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                    axs[0].text(x, y, f"{LCB_list[i]:.2f}", c='red', fontsize=6, ha='right', va='top')

                
            axs[0].legend(loc="lower left")
            axs[0].set_aspect('equal')


            # 第二个子图
            surf2 = axs[1].plot_surface(X_test_xx, X_test_yy, f(X_test).reshape(test_resolution), cmap='viridis', edgecolor='k', linewidth=0.5)
            # fig.colorbar(surf2, ax=axs[1], shrink=0.6)
            zmin = 0  # 设置 z 轴的最小值
            zmax = 0.3  # 设置 z 轴的最大值
            axs[1].set_zlim([zmin, zmax])
            # ax[0].set_title("Ground Truth\n$f(x, y) = x^2 + 2y^2$")
            # # ax[0].contourf(X_test_xx, X_test_yy, f(X_test).reshape(test_resolution))
            # ax[0].plot_surface(X_test_xx, X_test_yy, f(X_test).reshape(test_resolution), cmap='viridis', edgecolor='k', linewidth=0.5)
            # # for i, gp in enumerate(GPs):
            # ax[0].scatter(*gp.X_train_.T, marker=".", label=f"train points")

            # ax[0].legend(loc="lower left")


            # 第三个子图
            surf3 = axs[2].plot_surface(X_test_xx, X_test_yy, μ_test_2D, cmap='viridis', edgecolor='k', linewidth=0.5)
            # fig.colorbar(surf3, ax=axs[2], shrink=0.6)
            zmin = 0  # 设置 z 轴的最小值
            zmax = 0.3  # 设置 z 轴的最大值
            axs[2].set_zlim([zmin, zmax])
            # ax[1].set_title("Posterior Mean\n$\mu_{2|1}$")
            # ax[1].contourf(X_test_xx, X_test_yy, μ_test_2D)

            # 第4个子图
            surf4 = axs[3].plot_surface(X_test_xx, X_test_yy, phi_vals_2D, cmap='viridis', edgecolor='k', linewidth=0.5)
            fig.colorbar(surf4, ax=axs[3], pad=0.2, shrink=0.4)
            zmin = 0  # 设置 z 轴的最小值
            zmax = 0.3  # 设置 z 轴的最大值
            axs[3].set_zlim([zmin, zmax])
            # ax[2].set_title("Posterior Variance\n$\sigma^2_{2|1}$")
            # ax[2].contourf(X_test_xx, X_test_yy, σ_test_2D)

            # 设置标签和标题 
            axs[0].set_xlabel('X Label')
            axs[0].set_ylabel('Y Label')
            axs[0].set_title('Trajectory')
            
            axs[1].set_xlabel('X Label')
            axs[1].set_ylabel('Y Label')
            axs[1].set_zlabel('Z Label')
            axs[1].set_title('Ground Truth')

            axs[2].set_xlabel('X Label')
            axs[2].set_ylabel('Y Label')
            axs[2].set_zlabel('Z Label')
            axs[2].set_title('Mean value')
            
            axs[3].set_xlabel('X Label')
            axs[3].set_ylabel('Y Label')
            axs[3].set_zlabel('Z Label')
            axs[3].set_title('UCB')
            plt.title('Planel')
            plt.show()
            # plt.pause(0.01)
            # plt.clf()

    print("Done")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass