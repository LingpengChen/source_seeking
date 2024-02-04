## Define the source field
from scipy.stats import multivariate_normal
import numpy as np
from matplotlib import pyplot as plt


FOUND_SOURCE_THRESHOLD = 0.25 # to distinguish whether this is one source or two source
LCB_THRESHOLD = 0.1 # determine to seek which peak 
FIELD_SIZE_X = 10
FIELD_SIZE_Y = 10

SOURCE = np.array([[2,6], [8,7],[8,2], [5,6], [3,2]])
SOURCE = np.array([[2,6], [8,7],[8,2], [5,6], [3,2]])
SOURCE = np.array([[2,6], [8,7],[8,2], [5,6], [3,2]])
SOURCE = np.array([[2,6], [8,7],[8,2], [5,6], [3,2]])

source1 = multivariate_normal(SOURCE[0], 0.8*np.eye(2))
source2 = multivariate_normal(SOURCE[1], 0.9*np.eye(2))
source3 = multivariate_normal(SOURCE[2], 0.85*np.eye(2))
source4 = multivariate_normal(SOURCE[3], 0.9*np.eye(2))
source5 = multivariate_normal(SOURCE[4], 0.9*np.eye(2))

f = lambda x: source1.pdf(x) + 1.05*source2.pdf(x) + source3.pdf(x) + 1.1*source4.pdf(x) + source5.pdf(x)
source_value = f(SOURCE)

def sampling(x):
    σ_noise = 0.001
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    res = f(x) + np.random.normal(0, σ_noise, size=(x.shape[0],))
    # print(f(x), res, x.shape[0])
    return res

def find_source(setpoint, threshold=0.2):
    for coord in SOURCE:
        if np.linalg.norm(coord - setpoint) < threshold:
            return coord
    return None

if __name__ == '__main__':
    # source_value = sampling(SOURCE)
    # print(source_value)

    # setpoint = np.array([8, 7.14])  # 假设的机器人位置

    # index = find_source(setpoint)
    # print(index)  # 如果找到了匹配的坐标，这将打印出它的索引；否则打印 None
    FIELD_SIZE_X = 10
    FIELD_SIZE_Y = 10
    x_min = (0, 0)
    x_max = (0+FIELD_SIZE_X, 0+FIELD_SIZE_Y)
    
    test_resolution = [50, 50]
    X_test_x = np.linspace(x_min[0], x_max[0], test_resolution[0])
    X_test_y = np.linspace(x_min[1], x_max[1], test_resolution[1])
    X_test_xx, X_test_yy = np.meshgrid(X_test_x, X_test_y)
    X_test = np.vstack(np.dstack((X_test_xx, X_test_yy)))
    
    plt.contourf(X_test_xx, X_test_yy, f(X_test).reshape(test_resolution))
    
    plt.show()