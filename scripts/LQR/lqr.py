import numpy as np
import control
from double_integrator import DoubleIntegrator


# LQR 控制器
def lqr_gain(A, B, Q, R):
    """
    Compute the LQR feedback gain for continuous system
    """
    P, _, _ = control.care(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K

class DoubleIntegratorLQR(DoubleIntegrator):
    def __init__(self):
        super().__init__()
        # 定义Q和R权重矩阵
        self.Q = np.diag([10, 10, 1, 1])
        self.R = np.diag([1, 1])
        self.K = lqr_gain(self.A, self.B, self.Q, self.R)

    def control(self, x_desired, x_current):
        """
        LQR control law
        """
        error = x_current - x_desired
        u = -self.K @ error
        return u

# 使用LQR控制器
robot = DoubleIntegratorLQR()
x0 = np.array([0, 0, 10, 0])
robot.reset(x0)
x_desired = np.array([10, 10, 0, 0])

N = 200  # Number of steps
trajectory = [x0]
for _ in range(N):
    u = robot.control(x_desired, robot.state)
    x_next = robot.step(u)
    print(u)
    trajectory.append(x_next)

# 可以使用matplotlib绘制结果
import matplotlib.pyplot as plt

trajectory = np.array(trajectory)
plt.scatter(trajectory[:, 0], trajectory[:, 1], label='Robot Path', s=10, c='blue')
plt.scatter([10], [10], color='red', marker='*', label='Target')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.title('Double Integrator Path with LQR Control')
plt.grid(True)
plt.show()
