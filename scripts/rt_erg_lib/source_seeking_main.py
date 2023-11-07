import numpy as np
np.random.seed(10)

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, WhiteKernel
# from rt_erg_lib.controller import Controller

from utils import find_peak, calculate_wrmse
from controller import Controller

from vonoroi_utils import voronoi_neighbours
## Initilize environment
from IPython.display import clear_output

from scipy.spatial import Voronoi, voronoi_plot_2d
from environment_and_measurement import f, sampling, source, source_value # sampling function is just f with noise
                                                                        # while source and source_value are variables
COLORS = ['blue', 'green', 'red', 'purple', 'orange', 'black']  # 你可以添加更多颜色

# np.set_printoptions(threshold=np.inf, linewidth=np.inf, edgeitems=np.inf)
# print(μ_test.reshape(self.test_resolution))

def main():
        
    ## Define the source field
    FIELD_SIZE_X = 10
    FIELD_SIZE_Y = 10
    x_min = (0, 0)
    x_max = (0+FIELD_SIZE_X, 0+FIELD_SIZE_Y)

    ## Initialize GP (uni-GP)
    # if True:
    #     # Specify kernel with initial hyperparameter estimates
    #     def kernel_initial(
    #         σf_initial=1.0,         # covariance amplitude
    #         ell_initial=1.0,        # length scale
    #         σn_initial=0.1          # noise level
    #     ):
    #         return σf_initial**2 * RBF(length_scale=ell_initial) + WhiteKernel(noise_level=σn_initial)

    #     gp = GaussianProcessRegressor(
    #         kernel=kernel_initial(),
    #         n_restarts_optimizer=10
    #     )

    ## 用于采样的参数
    if True:
        # Predict points at uniform spacing to capture function
        
        test_resolution = [50, 50]
        X_test_x = np.linspace(x_min[0], x_max[0], test_resolution[0])
        X_test_y = np.linspace(x_min[1], x_max[1], test_resolution[1])
        X_test_xx, X_test_yy = np.meshgrid(X_test_x, X_test_y)
        X_test = np.vstack(np.dstack((X_test_xx, X_test_yy)))

    ## Initial robots
    # Give Robots Prior knowledge (all the same)
    if True:
        n_train = 0
        # Number of training points and testing points
        X_train=None
        y_train=None
        if (n_train):
            # Sample noisy observations (X1, y1) from the function for each of the GP components
            X_train = np.random.uniform(x_min, x_max, size=(n_train, 2))
            y_train = sampling(X_train)
            
    if True:
        robo_num = 3
        # robot_locations = [[3,3], [5, 4], [7,3]]
        robot_locations = [[1,2], [2, 2], [3,1]]
        Robots = []
        for index in range(robo_num):
            # initialize the thread
            instance = Controller(start_position=robot_locations[index], index=index, test_resolution=[50,50], field_size=[10,10]) # resolution is 50 * 50
            instance.receive_prior_knowledge(X_train, y_train)
            Robots.append(instance)

    
    rmse_values = []

    ## start source seeking!
    for iteration in range(100):
        
        print(iteration)
        
        ## 1 update vonoroi tessellation
        if True:
            # 1-1 find the neighbours of each robot
            robot_locations.clear()
            for i in range(robo_num):
                robot_locations.append(Robots[i].get_trajectory()[-1])
            neighbour_list = voronoi_neighbours(robot_locations)
            # 1-2 update voronoi cell info to each agents and exchange samples 
            exchange_dictionary_X = {}
            exchange_dictionary_y = {}
            
            for i in range(robo_num):
                temp_dictionary_X, temp_dictionary_y = Robots[i].voronoi_update(neighbour_list[i], robot_locations)
                for k, v in temp_dictionary_X.items():
                    # 如果键还不存在于 merged_dict 中，则创建一个新列表
                    if k not in exchange_dictionary_X:
                        exchange_dictionary_X[k] = []
                    # 将当前字典的值添加到合并字典中对应的键
                    exchange_dictionary_X[k].extend(v)
                    
                for k, v in temp_dictionary_y.items():
                    # 如果键还不存在于 merged_dict 中，则创建一个新列表
                    if k not in exchange_dictionary_y:
                        exchange_dictionary_y[k] = []
                    # 将当前字典的值添加到合并字典中对应的键
                    exchange_dictionary_y[k].extend(v)
            # 1-3 receive samples
            # print(exchange_dictionary)
            for i in range(robo_num):
                if i in exchange_dictionary_X:
                    Robots[i].receive_samples(exchange_dictionary_X[i], exchange_dictionary_y[i])
        
        ## 2. train and estimate!
        μ_estimation = np.zeros(test_resolution)
        ucb = np.zeros(test_resolution)
        # ucb = np.zeros(test_resolution)
        for i in range(robo_num):
            # μ_test, σ_test = Robots[i].gp_regresssion(ucb_coeff=0.5)
            μ_partial, ucb_partial = Robots[i].gp_regresssion(ucb_coeff=2)
            μ_estimation += μ_partial
            ucb += ucb_partial
            sources, lcb = Robots[i].get_estimated_source()
            # print(sources, lcb)
        # ucb miu estimation
        if 0:
            x = np.linspace(0, 49, 50)
            y = np.linspace(0, 49, 50)
            x, y = np.meshgrid(x, y)
            fig, ax = plt.subplots(1, 2,  subplot_kw={'projection': '3d'})
            ## 3. doing estimation based on the gp model trained in the last round
            ax[0] = fig.add_subplot(121, projection='3d')
            ax[0].plot_surface(x, y, μ_estimation, cmap='viridis')
            
            ax[1] = fig.add_subplot(122, projection='3d')
            ax[1].plot_surface(x, y, ucb, cmap='viridis')
            # ax[1] = fig.add_subplot(122, projection='3d')
            # ax[1].plot_surface(x, y, ucb, cmap='viridis')
            plt.show()
        
        
    
        
        # μ_test, σ_test = gp.predict(X_test, return_std=True)
        # μ_test_2D = μ_test.reshape(test_resolution)
        # σ_test_2D = σ_test.reshape(test_resolution)
        
        # # UCB!
        # phi_vals = μ_test + 0.5*σ_test
        # phi_vals_2D = phi_vals.reshape(test_resolution)
        
        ## 3. communication ck and phi_k and consensus
        ck_pack = {}
        phik_pack = {}
        for i in range(robo_num):
            ck_pack[i] = Robots[i].send_out_ck()
        for i in range(robo_num):
            Robots[i].receive_ck_consensus(ck_pack.copy()) 
            
        for i in range(robo_num):
            phik_pack[i] = Robots[i].send_out_phik()
        for i in range(robo_num):
            if i == 0 :
                ucb_changed = Robots[i].receive_phik_consensus(phik_pack.copy()) 
            else:
                Robots[i].receive_phik_consensus(phik_pack.copy()) 

        
        ## 4. Move and taking samples!
        for i in range(robo_num):
            if iteration < 5:
                setpts = Robots[i].get_nextpts(uniform = True)
            else:
                # setpts = Robots[i].get_nextpts(phi_vals = ucb)
                setpts = Robots[i].get_nextpts()
                
        
            # 4. Take samples and add to dataset
            # measurements = sampling(setpts) 
            # X_train = np.concatenate((X_train, setpts), axis=0)
            # y_train = np.concatenate((y_train, measurements), axis=0)
            
        
    ####################################################################
    ## Visualize
        SHOWN = True
        RMS_SHOW = False
        WRMSE = 1
        if RMS_SHOW:
            plt.ion()  # 开启interactive mode
            fig, ax = plt.subplots()
            if WRMSE:
                rmse = calculate_wrmse(μ_estimation, f(X_test).reshape(test_resolution))
                rmse_values.append(rmse)
                print(rmse)
                # 更新图表
                ax.clear()  # 清除旧的线条
                ax.plot(rmse_values)  # 绘制新的线条
                ax.set_title('WRMSE over iterations')
                ax.set_xlabel('Iteration')
                ax.set_ylabel('WRMSE')

                # 重绘图表
                fig.canvas.draw()
                fig.canvas.flush_events()
                clear_output(wait=True)  # 清除输出并显示新图
                
            else:
                rmse = np.sqrt(np.mean((μ_estimation - f(X_test).reshape(test_resolution)) ** 2))
                rmse_values.append(rmse)
                print(rmse)
                # 更新图表
                ax.clear()  # 清除旧的线条
                ax.plot(rmse_values)  # 绘制新的线条
                ax.set_title('RMSE over iterations')
                ax.set_xlabel('Iteration')
                ax.set_ylabel('RMSE')

                # 重绘图表
                fig.canvas.draw()
                fig.canvas.flush_events()
                clear_output(wait=True)  # 清除输出并显示新图
        
        if (iteration > 0 and iteration % 10 == 0 and SHOWN):
            sizes = 5  # 可以是一个数字或者一个长度为N的数组，表示每个点的大小              
            # # 设置图表的总大小
            fig, axs = plt.subplots(1, 5, figsize=(24, 10), subplot_kw={'projection': '3d'})
            axs[0].remove()  # 移除第一个子图的3D投影
            axs[0] = fig.add_subplot(1, 5, 1)  # 添加一个2D子图

            ## Start plotting
            
            ## 在第1个子图上绘制2D图
            if True:   
                contour = axs[0].contourf(X_test_xx, X_test_yy, f(X_test).reshape(test_resolution))

                # 1-1) plot Voronoi tessellation
                vor = Voronoi(robot_locations)
                voronoi_plot_2d(vor, ax=axs[0], show_vertices=False)
                axs[0].set_xlim([0, 10])  # 你可以根据需要调整这些值
                axs[0].set_ylim([0, 10])  
                
                # 1-2) trajectory
                for i in range(robo_num):
                    color = COLORS[i % len(COLORS)]
                    trajectory = np.array(Robots[i].trajectory)
                    axs[0].plot(trajectory[:, 0], trajectory[:, 1], color=color, zorder=1)  # 设置线条颜色
                    axs[0].scatter(trajectory[:, 0], trajectory[:, 1],s=sizes, c=color, zorder=2)
                # axs[0].scatter(Robots[1].samples_X[:, 0], Robots[1].samples_X[:, 1], s=sizes, color='orange', zorder=3)

                # 1-3) sources ground truth
                x_coords = source[:, 0]
                y_coords = source[:, 1]
                axs[0].scatter(x_coords, y_coords, s=10, c='black', marker='x', zorder=3)
                for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                    axs[0].text(x, y, f"{source_value[i]:.2f}", fontsize=6, ha='right', va='bottom')
                
                # 1-4) estimated sources
                # if (len(peaks)!=0):
                #     peaks = np.array(peaks)
                #     x_coords = FIELD_SIZE_X * peaks[:, 1] / test_resolution[0]
                #     y_coords = FIELD_SIZE_Y * peaks[:, 0] / test_resolution[1]
                #     axs[0].scatter(x_coords, y_coords, s=10, c='red', marker='x', zorder=3)
                #     for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                #         axs[0].text(x, y, f"{LCB_list[i]:.2f}", c='red', fontsize=6, ha='right', va='top')

                # axs[0].legend(loc="lower left")
                axs[0].set_aspect('equal')


            # 第2个子图 ground truth
            surf2 = axs[1].plot_surface(X_test_xx, X_test_yy, f(X_test).reshape(test_resolution), cmap='viridis', edgecolor='k', linewidth=0.5)
            zmin = 0  # 设置 z 轴的最小值
            zmax = 0.3  # 设置 z 轴的最大值
            axs[1].set_zlim([zmin, zmax])


            # 第3个子图 estimated distribution
            surf3 = axs[2].plot_surface(X_test_xx, X_test_yy, μ_estimation, cmap='viridis', edgecolor='k', linewidth=0.5)
            zmin = 0  # 设置 z 轴的最小值
            zmax = 0.3  # 设置 z 轴的最大值
            axs[2].set_zlim([zmin, zmax])
  
            # 第4个子图
            
            surf4 = axs[3].plot_surface(X_test_xx, X_test_yy, ucb, cmap='viridis', edgecolor='k', linewidth=0.5)
            fig.colorbar(surf4, ax=axs[3], pad=0.2, shrink=0.4)
            
            surf5 = axs[4].plot_surface(X_test_xx, X_test_yy, ucb_changed, cmap='viridis', edgecolor='k', linewidth=0.5)
            fig.colorbar(surf5, ax=axs[4], pad=0.2, shrink=0.4)
            # zmin = 0  # 设置 z 轴的最小值
            # zmax = 0.3  # 设置 z 轴的最大值
            # axs[3].set_zlim([zmin, zmax])

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
            
            axs[4].set_xlabel('X Label')
            axs[4].set_ylabel('Y Label')
            axs[4].set_zlabel('Z Label')
            axs[4].set_title('UCB_changed')
            plt.show()
            # plt.pause(0.01)
            # plt.clf()

    print("Done")


if __name__ == '__main__':
    main()
   