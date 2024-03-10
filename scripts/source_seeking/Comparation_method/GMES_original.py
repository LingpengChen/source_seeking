import numpy as np
import sklearn.gaussian_process.kernels as kernels
from tqdm import tqdm
from matplotlib import cm

from source_seeking_module.bayesian_optimization import BayesianOptimizationCentralized
from source_seeking_module.benchmark_functions_2D import *
from environment.environment_and_measurement import Environment

COLORS = ['blue', 'green', 'red', 'purple', 'orange', 'black']  # 你可以添加更多颜色

# Set seed
np.random.seed(0)


# Bayesian optimization object
environment = Environment( 1 )

n_workers = 3
x0 = np.array([[1,2], [2,2], [3, 1]])
# n_workers = 1
# x0 = np.array([[1,2]])
BO = BayesianOptimizationCentralized(domain=np.array([[0, 10], [0, 10]]),
                            n_workers=n_workers,
                            kernel='Matern',
                            acquisition_function='es',
                            policy='greedy',
                            regularization=None,
                            regularization_strength=0.01,
                            pending_regularization=None,
                            pending_regularization_strength=0.01,
                            grid_density=100,
                            env = environment
                            )


# Optimize
BO.initialize(x0, environment.sampling(x0), n_pre_samples=20)

# start experiments!
query = None
for iteration in tqdm(range(101), position = 1, leave = None):

    ## Step 1: Train
    # BO.train_gp(query)
    BO.train_gp_query(query_x=query, query_y=environment.sampling([query]))
    ## Step 2: query
    # find next query # array([[5.488135 , 7.1518936]], dtype=float32)
    query = BO.find_next_query(iteration)
    # Plot optimization step
    if iteration % 5 ==0:
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        first_param_grid = np.linspace(0, 10, 100)
        second_param_grid = np.linspace(0, 10, 100)
        X_plot, Y_plot = np.meshgrid(first_param_grid, second_param_grid, indexing='ij')
        x = np.array(BO.X)
        y = np.array(BO.Y)
        mu, std = BO.model.predict(BO._grid, return_std=True)
        
        # Objective plot
        N = 100
        Y_obj = [environment.get_gt(i) for i in BO._grid]
        clev1 = np.linspace(min(Y_obj), max(Y_obj),N)
        cp1 = axs[0].contourf(X_plot, Y_plot, np.array(Y_obj).reshape(X_plot.shape), clev1,  cmap = cm.coolwarm)

        # trajectory
        for i in range(BO.n_workers):
            color = COLORS[i % len(COLORS)]
            trajectory = np.array([i.tolist() for i in BO.X[i+20::BO.n_workers]])                        # trajectory = np.array(Robots[i].trajectory)
            axs[0].plot(trajectory[:, 0], trajectory[:, 1], color=color, zorder=1)  
            axs[0].scatter(trajectory[:, 0], trajectory[:, 1], c=color, zorder=2)

        
        cp2 = axs[1].contourf(X_plot, Y_plot, mu.reshape(X_plot.shape),  cmap = cm.coolwarm)

        axs[0].set_xlabel('X Label')
        axs[0].set_ylabel('Y Label')
        axs[0].set_title('Trajectory and Ground Truth')
        
        axs[1].set_xlabel('X Label')
        axs[1].set_ylabel('Y Label')
        axs[1].set_title('Mean value')

        plt.show()


