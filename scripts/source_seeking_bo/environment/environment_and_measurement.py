## Define the source field
from scipy.stats import multivariate_normal
import numpy as np
from matplotlib import pyplot as plt

DEBUG = False
FIELD_SIZE = [10, 10]

CAM_FOV = 0.4 # to distinguish whether this is one source or two source
SRC_MUT_D_THRESHOLD = 1

LCB_THRESHOLD = 0.15  
FIELD_SIZE_X = 10
FIELD_SIZE_Y = 10

USE_BO = True
BO_RADIUS = 0.5

CTR_MAG_DETERMIN_STUCK = 0.15
STUCK_PTS_THRESHOLD = 0.1 # smaller than this threshold are considered as the same stuck pt

SOURCES_case = np.array([[[2,6], [8,7],[8,2], [5,6], [3,2]],  # 0
[[3.0, 4.5], [6.5, 2.5], [2.5, 8.0], [5.5, 7.0], [8.0, 7.0]] ,
[[8.0, 2.5], [2.5, 5.5], [7.5, 5.5], [4.5, 2.0], [6.0, 8.0]] , 
[[8.0, 8.0], [6.0, 5.5], [6.0, 2.5], [2.0, 2.0], [3.5, 8.0]] , #3
[[4.0, 7.5], [3.5, 2.0], [7.5, 7.0], [2.0, 5.0], [7.5, 2.0]] ,
[[6.0, 4.5], [3.5, 2.0], [7.0, 7.5], [2.5, 7.0], [8.0, 2.5]] , 
[[3.0, 7.0], [3.5, 3.0], [8.0, 4.0], [6.0, 6.0], [7.5, 8.0]] , #6
[[4.5, 5.0], [2.5, 3.0], [8.0, 6.5], [7.5, 2.0], [3.0, 8.0]] , 
[[5.5, 6.0], [2.5, 3.0], [6.5, 2.0], [3.5, 8.0], [8.0, 5.0]] , 
[[5.0, 6.5], [2.0, 3.5], [8.0, 8.0], [2.0, 8.0], [5.0, 2.5]] ,   #9
                        #  [[2,6], [8,7],[8,2], [5,8], [3,2]], #11
# [[4.5, 2.0], [8.0, 7.0], [7.5, 3.5], [5.5, 7.0], [2.0, 6.0]] , #6
])

ROBOT_INIT_LOCATIONS_case = [[[1,2], [2, 2], [3,1]],
                            [[1,8], [2, 8], [3,9]],
                            [[9,8], [8, 8], [7,9]],
                            [[9,2], [8, 2], [7,1]]]

class Environment():    # class for robot to interact with the environment
    def __init__(self, source_case_index):
        self.SOURCES = SOURCES_case[source_case_index]
        self.SOURCE_SET = {tuple(item) for item in self.SOURCES}
        self.field_size = FIELD_SIZE
        n=0.6
        def get_f(sources):
            source1 = multivariate_normal(sources[0], 0.8*n*np.eye(2))
            source2 = multivariate_normal(sources[1], 0.9*n*np.eye(2))
            source3 = multivariate_normal(sources[2], 0.85*n*np.eye(2))
            source4 = multivariate_normal(sources[3], 0.9*n*np.eye(2))
            source5 = multivariate_normal(sources[4], 0.9*n*np.eye(2))

            f = lambda x: source1.pdf(x) + 1.05*source2.pdf(x) + source3.pdf(x) + 1.1*source4.pdf(x) + source5.pdf(x)
            return f
        self.f = get_f(self.SOURCES)
        self.SOURCE_VALUE = self.f(self.SOURCES)
    
    def get_gt(self, x):
        return self.f(x)
    
    def sampling(self, x):
        σ_noise = 0.001
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        res = self.f(x) + np.random.normal(0, σ_noise, size=(x.shape[0],))
        # print(f(x), res, x.shape[0])
        return res

    def find_source(self, setpoint):
        for coord in self.SOURCES:
            if np.linalg.norm(coord - setpoint) < CAM_FOV:
                return coord
        return None


if __name__ == '__main__':
    # source_value = sampling(SOURCE)
    FIELD_SIZE_X = 10
    FIELD_SIZE_Y = 10
    x_min = (0, 0)
    x_max = (0+FIELD_SIZE_X, 0+FIELD_SIZE_Y)
    test_resolution = [50, 50]
    X_test_x = np.linspace(x_min[0], x_max[0], test_resolution[0])
    X_test_y = np.linspace(x_min[1], x_max[1], test_resolution[1])
    X_test_xx, X_test_yy = np.meshgrid(X_test_x, X_test_y)
    X_test = np.vstack(np.dstack((X_test_xx, X_test_yy)))

    for i, sources in enumerate(SOURCES_case):
        env = Environment(i)


        # # 设置图像的长宽比为一致
        plt.gca().set_aspect('equal', adjustable='box')

        plt.contourf(X_test_xx, X_test_yy, env.f(X_test).reshape(test_resolution), cmap='coolwarm',  edgecolor='none', levels=100)
        plt.xticks([])
        plt.yticks([])
        # 
        plt.show()
        fig = plt.figure(figsize=(40, 20))
        
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1,1,0.1])  # Aspect ratio is 1:1:0.2, making Z axis changes appear flatter

        ax.plot_surface(X_test_xx, X_test_yy, env.f(X_test).reshape(test_resolution), cmap='coolwarm', edgecolor='none', rcount=100, ccount=100, linewidth=0.5)
        ax.view_init(elev=30, azim=45)  # elev是仰角，azim是方位角
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # 隐藏所有轴标签
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')

        # 隐藏网格线
        ax.grid(False)

        ax.set_axis_off()


        # 隐藏轴背景
        ax.set_facecolor((0,0,0,0))  # 设置透明背景

        # 隐藏边框
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        plt.show()
        
        
        break
    