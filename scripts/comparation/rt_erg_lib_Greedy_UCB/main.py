import warnings
warnings.filterwarnings("ignore")

import numpy as np
np.random.seed(10)

from matplotlib import pyplot as plt
plt.rcParams['font.size'] = 20  # 设置全局字体大小为12

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
from environment_and_measurement import f, sampling, SOURCE, source_value # sampling function is just f with noise
                                                                        # while SOURCE and source_value are variables
COLORS = ['blue', 'green', 'red', 'purple', 'orange', 'black']  # 你可以添加更多颜色
SOURCE_set = {tuple(item) for item in SOURCE}
# np.set_printoptions(threshold=np.inf, linewidth=np.inf, edgeitems=np.inf)
# print(μ_test.reshape(self.test_resolution))

def unique_list(redundant_list):
    # 将内部列表转换成元组
    tuple_list = [tuple(list(item)) for item in redundant_list]
    # 使用集合去除重复元素，并转换回列表
    unique_list = [list(item) for item in set(tuple_list)]
    return unique_list

def main():
        
    ## Define the source field
    FIELD_SIZE_X = 10
    FIELD_SIZE_Y = 10
    x_min = (0, 0)
    x_max = (0+FIELD_SIZE_X, 0+FIELD_SIZE_Y)

    ## Initialize GP (uni-GP)
 
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
    SHOWN =   True
    RMS_SHOW = not SHOWN
    end = False
    WRMSE = 1
    if RMS_SHOW:
        plt.ion()  # 开启interactive mode
        fig, ax = plt.subplots()
    ## start source seeking!
    for iteration in range(300):
        
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
        peaks = []
        UCB_list = []
        coeff = 0
        if iteration < 100:
            coeff=4*(1-iteration/100)    
            
        for i in range(robo_num):
            # μ_test, σ_test = Robots[i].gp_regresssion(ucb_coeff=0.5)
            μ_partial, ucb_partial = Robots[i].gp_regresssion(ucb_coeff=2)
            μ_estimation += μ_partial
            ucb += ucb_partial  
            sources, ucb_value = Robots[i].get_estimated_source()
            peaks += sources
            UCB_list += ucb_value 
        
        ## 3. communication ck and phi_k and consensus
        ck_pack = {}
        phik_pack = {}
        found_source = []
        for i in range(robo_num):
            found_source += Robots[i].send_out_source_cord()
            ck_pack[i] = Robots[i].send_out_ck()
            phik_pack[i] = Robots[i].send_out_phik()
        found_source = unique_list(found_source)
        # 将子列表转换为元组，然后转换为集合
        found_source_set = {tuple(item) for item in found_source}
        if found_source_set == SOURCE_set:
            end = True

        for i in range(robo_num):
            Robots[i].receive_ck_consensus(ck_pack.copy()) 
            Robots[i].receive_source_cord(found_source)
            ucb_changed = Robots[i].receive_phik_consensus(phik_pack.copy()) 
        
        ## 4. Move and taking samples!
        targets = []
        for i in range(robo_num):
            setpts, target = Robots[i].get_nextpts(control_mode = "UCB_greedy") 
            targets.append(target)
            
      
    ####################################################################
    ## Visualize
        
        if (iteration in [30, 45, 60, 90, 110, 51, 74, 106]) :
            sizes = 5  # 可以是一个数字或者一个长度为N的数组，表示每个点的大小              
            # # 设置图表的总大小
         
            fig, ax = plt.subplots(figsize=(10, 10))

            ## Start plotting
            
            ## 在第1个子图上绘制2D图
            contour = plt.contourf(X_test_xx, X_test_yy, f(X_test).reshape(test_resolution))

            # 1-1) plot Voronoi tessellation
            # vor = Voronoi(robot_locations)
            # voronoi_plot_2d(vor, ax=ax, show_vertices=False)
            ax.set_xlim([0, 10])  # 你可以根据需要调整这些值
            ax.set_ylim([0, 10])  
            
            # 1-2) trajectory
            for i in range(robo_num):
                color = COLORS[i % len(COLORS)]
                trajectory = np.array(Robots[i].trajectory)
                ax.plot(trajectory[:, 0], trajectory[:, 1], lw =3, color=color, zorder=1)  # 设置线条颜色
                ax.scatter(trajectory[:, 0], trajectory[:, 1],s=50, c=color, zorder=2)
            # ax.scatter(Robots[1].samples_X[:, 0], Robots[1].samples_X[:, 1], s=sizes, color='orange', zorder=3)

            # 1-3) sources ground truth
            x_coords = SOURCE[:, 0]
            y_coords = SOURCE[:, 1]
            ax.scatter(x_coords, y_coords, s=30, c='black', marker='x', zorder=5)
            # for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            #     ax.text(x, y, f"{source_value[i]:.2f}", fontsize=6, ha='right', va='bottom')
            
         
            # 设置标签和标题 
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
        
            plt.show()
            file_name = "./record/" + str(iteration) + ".png"
            fig.savefig(file_name, bbox_inches='tight')  # 保存图像为PDF格式，去掉空白边界

            if iteration>200:
                plt.pause(1000)
                break
            # plt.pause(0.01)
            # plt.clf()


    print("Done")


if __name__ == '__main__':
    main()
   