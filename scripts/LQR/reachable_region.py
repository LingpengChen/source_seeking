import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from gym.spaces import Box
from double_integrator import DoubleIntegrator
from matplotlib.patches import Polygon

class DoubleIntegratorVisualizer(DoubleIntegrator):

    def plot_reachable_area(self, current_state, time_step=1):
        # 接受当前状态[current_x, current_y, velocity_x, velocity_y]
        # 和时间步长time_step作为参数
        self.reset(current_state)

        # 设定控制输入的最大最小值
        u_min, u_max = self.action_space.low, self.action_space.high
        ux_min, uy_min = u_min[0], u_min[1]
        ux_max, uy_max = u_max[0], u_max[1]
        # ux_min, uy_min, ux_max, uy_max
        # 使用最大最小控制输入来确定下一步的状态
        next_states = []
        
        possible_ctr = [[u_min[0], u_min[1]], [u_min[0], u_max[1]], [u_max[0], u_max[1]], [u_max[0], u_min[1]]]
        for i in len(possible_ctr):
            # 计算下一状态
            [ux, uy] = possible_ctr[i]
            self.step(np.array([ux, uy]))
            next_state = self.step(np.array([0, 0]))
            first_sample_X = next_state[:2]
            
            self.reset(current_state)
            print("next_state", next_state)
            next_states.append(next_state[:2])

        # 重置状态
        # 创建四边形表示可达区域
        reachable_area = Polygon(next_states, alpha=0.5, color='green')

        # 绘制当前状态和可达区域
        fig, ax = plt.subplots()
        ax.add_patch(reachable_area)
        ax.plot(current_state[0], current_state[1], 'ro')  # 当前位置
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_title('Reachable area after {} second(s)'.format(time_step))
        plt.show()

if __name__ == '__main__':    # 使用新的子类
    visualizer = DoubleIntegratorVisualizer()
    # 设置当前状态
    current_state = np.array([0, 0, 0, 1])
    # 可视化可达区域
    visualizer.plot_reachable_area(current_state)
