import cvxpy as cp
import numpy as np
from double_integrator import DoubleIntegrator
import matplotlib.pyplot as plt

class MPCController:
    def __init__(self, model, horizon=5, Q=None, R=None):
        self.model = model
        self.horizon = horizon
        self.u = cp.Variable((2, horizon))
        self.x = cp.Variable((4, horizon+1))
        self.dt = model.dt
        
        # Weights for the cost function
        if Q is None:
            self.Q =np.diag([1, 1, 0, 0])
        else:
            self.Q = Q
        
        if R is None:
            self.R = 0.01*np.eye(2)
        else:
            self.R = R

    def control(self, x0, x_target):
        cost = 0
        constraints = []
        for t in range(self.horizon):
            cost += cp.quad_form(self.x[:, t] - x_target, self.Q)
            cost += cp.quad_form(self.u[:, t], self.R)
            
            constraints += [self.x[:, t+1] == self.x[:, t] + (self.model.A @ self.x[:, t] + self.model.B @ self.u[:, t]) * self.dt]
            constraints += [self.model.action_space.low <= self.u[:, t], 
                            self.u[:, t] <= self.model.action_space.high]
            
        constraints += [self.x[:, 0] == x0]
        
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()
        
        if problem.status == cp.OPTIMAL:
            return self.u[:, 0].value
        else:
            return np.zeros(2)

# Test the MPC Controller
robot = DoubleIntegrator()
mpc = MPCController(robot, horizon=10)

x0 = np.array([0., 0., 0.3, -0.1])
x_target = np.array([1., 1., 0., 0.])
trajectory = [x0]

for i in range(30):
    u = mpc.control(trajectory[-1], x_target)
    robot.reset(trajectory[-1])
    x_next = robot.step(u)
    delta = x_next[:2] - x_target[:2]
    if np.sqrt(np.dot(delta, delta)) < 0.05:
        print(i)
        break
    print(u)
    print(robot.state)
    trajectory.append(x_next)
    
trajectory = np.array(trajectory)

# Now, you can plot the trajectory
plt.scatter(trajectory[:, 0], trajectory[:, 1], label='Robot Path', s=10, c='blue')
plt.scatter([1], [1], color='red', marker='*', label='Target')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.title('Double Integrator Path with MPC Control')
plt.grid(True)
plt.show()
