import cvxpy as cp
import numpy as np
from double_integrator import DoubleIntegrator
import matplotlib.pyplot as plt

class MPCController:
    def __init__(self, model, horizon=10, Q=1, R=0.01):
        self.model = model
        self.horizon = horizon
        self.u = cp.Variable((2, horizon))
        self.x = cp.Variable((4, horizon+1))
        self.dt = model.dt
        
        # Weights for the cost function
        if Q is None:
            self.Q =np.diag([1, 1, 0, 0])
        else:
            self.Q = np.diag([Q, Q, 0, 0])
        
        if R is None:
            self.R = 0.001*np.eye(2)
        else:
            self.R = R*np.eye(2)

    def __call__(self, x0, x_target):
        if not isinstance(x_target, np.ndarray):
            x_target = np.array(x_target)
        if x_target.shape == (2,):
            x_target = np.pad(x_target, (0, 2), 'constant', constant_values=(0,))
        cost = 0
        constraints = []
        for t in range(self.horizon):
            cost += cp.quad_form(self.x[:, t] - x_target, self.Q)
            cost += cp.quad_form(self.u[:, t], self.R)
            
            constraints += [self.x[:, t+1] == self.x[:, t] + (self.model.A @ self.x[:, t] + self.model.B @ self.u[:, t]) * self.dt]
            constraints += [((self.model.A @ self.x[:, t] + self.model.B @ self.u[:, t]) * self.dt) <= 0.05]
            constraints += [cp.norm(self.u[:, t], 2) <= 1]
            
        constraints += [self.x[:, 0] == x0]
        
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()
        
        if problem.status == cp.OPTIMAL:
            return self.u[:, 0].value
        else:
            return np.zeros(2)

if __name__ == '__main__':
    # Test the MPC Controller
    robot = DoubleIntegrator()
    mpc = MPCController(robot, horizon=10)

    x0 = np.array([0.163, 0.429, 0.077, 0.627])
    x_target = np.array([0.1848, 0.5448])
    trajectory = [x0]

    for i in range(30):
        u = mpc(trajectory[-1], x_target)
        robot.reset(trajectory[-1])
        x_next = robot.step(u)
        delta = x_next[:2] - x_target[:2]
        # if np.sqrt(np.dot(delta, delta)) < 0.05:
        #     print(i)
        #     break
        print(u)
        print(robot.state)
        trajectory.append(x_next)
        
    trajectory = np.array(trajectory)

    # Now, you can plot the trajectory
    plt.scatter(trajectory[:, 0], trajectory[:, 1], label='Robot Path', s=10, c='blue')
    plt.scatter(x_target[0], x_target[1], color='red', marker='*', label='Target')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.title('Double Integrator Path with MPC Control')
    plt.grid(True)
    plt.show()
