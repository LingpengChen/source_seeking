## Define the source field
from scipy.stats import multivariate_normal
import numpy as np

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

def sampling(x):
    σ_noise = 0.001
    return f(x) + np.random.normal(0, σ_noise, size=(x.shape[0],))

if __name__ == '__main__':
    source_value = sampling(source)

    print(source_value)