from scipy.stats import multivariate_normal
import numpy as np

CAM_FOV = 0.4

class Environment():    # class for robot to interact with the environment
    def __init__(self, field_size, map_resolution):
        self.SOURCES = np.array([[2,6], [8,7],[8,2], [5,6], [3,2]])
        self.SOURCE_SET = {tuple(item) for item in self.SOURCES}
        self.map_resolution = map_resolution
        self.unit = field_size/map_resolution
        
        def get_f(sources):
            n=2
            source1 = multivariate_normal(sources[0], 0.8*n*np.eye(2))
            source2 = multivariate_normal(sources[1], 0.9*n*np.eye(2))
            source3 = multivariate_normal(sources[2], 0.85*n*np.eye(2))
            source4 = multivariate_normal(sources[3], 0.9*n*np.eye(2))
            source5 = multivariate_normal(sources[4], 0.9*n*np.eye(2))

            f = lambda x: source1.pdf(x) + 1.05*source2.pdf(x) + source3.pdf(x) + 1.1*source4.pdf(x) + source5.pdf(x)
            return f
        self.f = get_f(self.SOURCES)
        self.SOURCE_VALUE = self.f(self.SOURCES)
    
    def get_H(self, x):
        H_k = np.zeros((3, self.map_resolution**2))
        for i, cord in enumerate(x):
            row = (cord[1]-self.unit/2)/self.unit
            col = (cord[0]+self.unit/2)/self.unit 
            index = int(col + row * self.map_resolution) - 1
            H_k[i][index] = 1
        return H_k
        
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
    
        