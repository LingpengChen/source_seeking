## Define the source field
from scipy.stats import multivariate_normal
import numpy as np

FOUND_SOURCE_THRESHOLD = 0.25 # to distinguish whether this is one source or two source
# UCB_COEFF = 2 # ES
# UCB_COEFF = 2 # greedy UCB
UCB_COEFF = 1 # greedy MI

LCB_THRESHOLD = 0.12 # determine to seek which peak 
FIELD_SIZE_X = 10
FIELD_SIZE_Y = 10

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

    setpoint = np.array([8, 7.14])  # 假设的机器人位置

    index = find_source(setpoint)
    print(index)  # 如果找到了匹配的坐标，这将打印出它的索引；否则打印 None