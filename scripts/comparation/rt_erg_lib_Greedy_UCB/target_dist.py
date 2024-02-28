import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

class TargetDist(object):
    '''
    This is going to be a test template for the code,
    eventually a newer version will be made to interface with the
    unity env
    '''

    def __init__(self, num_nodes=2, num_pts=50):

        self.num_pts = num_pts
        grid = np.meshgrid(*[np.linspace(0, 1, num_pts) for _ in range(2)])
        self.grid = np.c_[grid[0].ravel(), grid[1].ravel()]

        self.means = [npr.uniform(0.0, 1.0, size=(2,))
                            for _ in range(num_nodes)]
        # self.means = [np.array([0.2, 0.2]), np.array([0.7,0.7])]
        # self.vars  = [np.array([0.1,0.1])**2, np.array([0.1,0.1])**2]

        self.vars  = [npr.uniform(0.1, 0.3, size=(2,))**2
                            for _ in range(num_nodes)]

        self.grid_vals = self.__call__(self.grid)

    def get_grid_spec(self):
        xy = []
        for g in self.grid.T:
            xy.append(
                np.reshape(g, newshape=(self.num_pts, self.num_pts))
            )
        return xy, self.grid_vals.reshape(self.num_pts, self.num_pts)


    def __call__(self, x):
        assert len(x.shape) > 1, 'Input needs to be a of size N x n'
        assert x.shape[1] == 2, 'Does not have right exploration dim'

        val = np.ones(x.shape[0])
        # for m, v in zip(self.means, self.vars):
        #     innerds = np.sum((x-m)**2 / v, 1)
        #     val += np.exp(-innerds/2.0)# / np.sqrt((2*np.pi)**2 * np.prod(v))
        
        # normalizes the distribution
        val /= np.sum(val)
        return val

if __name__ == '__main__':


    t_dist      = TargetDist(2)
    xy, vals = t_dist.get_grid_spec()

    plt.figure(1)
    plt.title('ergodic coverage')
    plt.contourf(*xy, vals, levels=10)
    plt.show()