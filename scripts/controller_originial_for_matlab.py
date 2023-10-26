import sys
sys.path.append('./rt_erg_lib/')
from rt_erg_lib.double_integrator import DoubleIntegrator
from rt_erg_lib.ergodic_control import RTErgodicControl
from rt_erg_lib.target_dist import TargetDist
from rt_erg_lib.target_dist_t import TargetDist_t

from rt_erg_lib.utils import convert_phi2phik, convert_ck2dist, convert_traj2ck, convert_phik2phi
import numpy as np
from scipy.io import loadmat

import array

class Controller(object): # python (x,y) therefore col index first, row next
    # robot state + controller !
    def __init__(self, start_position, row=40, col=20):
        self.row = row
        self.col = col
        self.location = [start_position[0]/self.col, start_position[1]/self.row]
        grid_2_r_w = np.meshgrid(np.linspace(0, 1, int(self.col)), np.linspace(0, 1, int(self.row)))
        self.grid = np.c_[grid_2_r_w[0].ravel(), grid_2_r_w[1].ravel()]
        
        self.robot_dynamic = DoubleIntegrator() # robot controller system
        self.model = DoubleIntegrator()
        
        self.robot_state     = np.zeros(self.robot_dynamic.observation_space.shape[0])
        self.robot_state[:2] = np.array(self.location)
        self.robot_dynamic.reset(self.robot_state)
        
        self.erg_ctrl    = RTErgodicControl(self.model, horizon=15, num_basis=5, batch_size=-1)

        self.pre_state = None
        
        print("Controller Succcessfully Initialized! Initial position is: ", start_position)
    
   
     
    def get_nextpts(self, phi_vals): 
        sample_steps = 3
        setpoints = []
        
        # setting the phik on the ergodic controller
        phi_vals = np.array(phi_vals)
        phi_vals /= np.sum(phi_vals)

        self.erg_ctrl.phik = convert_phi2phik(self.erg_ctrl.basis, phi_vals, self.grid)

        i = 0
        while (i < sample_steps):
            i += 1
            ctrl = self.erg_ctrl(self.robot_state)
            self.robot_state = self.robot_dynamic.step(ctrl)

            temp_state = [ int(self.robot_state[0]*self.col), int(self.robot_state[1]*self.row) ]
            if (self.pre_state != None) and (self.pre_state == temp_state):
                sample_steps += 1
                self.pre_state = temp_state
                continue
            self.pre_state = temp_state
            # cord = [round(self.robot_state[0]*self.col), round(self.robot_state[1]*self.row)]
            # if pre_cord != cord:  
            #     setpoints.append([round(self.robot_state[0]*self.col), round(self.robot_state[1]*self.row) ])
            # pre_cord = cord  
                  
            setpoints.append(temp_state)
            # plt.scatter(self.robot_state[0], self.robot_state[1])
            # plt.pause(0.001)  # 暂停绘图并刷新窗口
        
        setpoints = np.array(setpoints)
        return setpoints
        
   