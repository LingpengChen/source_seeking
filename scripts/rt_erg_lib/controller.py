import sys
# sys.path.append('./rt_erg_lib/')
from double_integrator import DoubleIntegrator
from ergodic_control import RTErgodicControl
from target_dist import TargetDist
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi, distance
import matplotlib.pyplot as plt

from utils import convert_phi2phik, convert_ck2dist, convert_traj2ck, convert_phik2phi
import numpy as np
from scipy.io import loadmat

import rospy
from grid_map_msgs.msg import GridMap
from source_seeking.msg import Ck
from environment_and_measurement import f, sampling, source, source_value # sampling function is just f with noise

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

def kernel_initial(
            σf_initial=1.0,         # covariance amplitude
            ell_initial=1.0,        # length scale
            σn_initial=0.1          # noise level
        ):
            return σf_initial**2 * RBF(length_scale=ell_initial, length_scale_bounds=(0.5, 2)) + WhiteKernel(noise_level=σn_initial)

class Controller(object): # python (x,y) therefore col index first, row next
    # robot state + controller !
    def __init__(self, start_position, index, test_resolution = [50,50], field_size = [10,10]):
        
        ## robot index
        self.index = index
        self.agent_name = str(index)
        
        ## robot node and communication

            # # send out phik
            # self._phik_msg    = Ck()
            # self._phik_msg.name = self.agent_name
            # self._phik_pub = rospy.Publisher('phik_link', Ck, queue_size=1)
            # # receive phik
            # rospy.Subscriber('phik_link', Ck, self._phik_link_callback)
            
            # # send out ck
            # self._ck_msg    = Ck()
            # self._ck_msg.name = self.agent_name
            # self._ck_pub = rospy.Publisher('ck_link', Ck, queue_size=1)
            # # send out ck
            # rospy.Subscriber('ck_link', Ck, self._ck_link_callback)
            # self._ck_dict   = {}
        
        ## field information
        # test_resolution
        self.test_resolution = test_resolution # [50, 50]
        # real size
        self.field_row = field_size[0] # 10
        self.field_col = field_size[1]
        
        grid_2_r_w = np.meshgrid(np.linspace(0, 1, int(test_resolution[0])), np.linspace(0, 1, int(test_resolution[1])))
        self.grid = np.c_[grid_2_r_w[0].ravel(), grid_2_r_w[1].ravel()] #(2500,2)
      
        X_test_x = np.linspace(0, field_size[0], test_resolution[0])
        X_test_y = np.linspace(0, field_size[1], test_resolution[1])
        X_test_xx, X_test_yy = np.meshgrid(X_test_x, X_test_y)
        self.X_test = np.vstack(np.dstack((X_test_xx, X_test_yy))) #(50,50)
        
        ## ROBOT information
        # robot dynamics
        self.robot_dynamic = DoubleIntegrator() # robot controller system
        self.robot_state     = np.zeros(self.robot_dynamic.observation_space.shape[0])
        # robot initial location
        self.robot_state[:2] = np.array([start_position[0]/self.field_row, start_position[1]/self.field_col])
        self.robot_dynamic.reset(self.robot_state)
        self.Erg_ctrl    = RTErgodicControl(DoubleIntegrator(), weights=0.01, horizon=15, num_basis=10, batch_size=-1)

        # samples 
        self.samples_X = np.array([start_position])
        self.samples_Y = np.array([f(start_position)])
        
        self.trajectory = [start_position]
        
        ## ROBOT neigibours
        self.neighbour = None  
        self.responsible_region = np.zeros(self.test_resolution)
        
        print("Controller Succcessfully Initialized! Initial position is: ", start_position)
        self.sent_samples = []
        
        ## GP
        self.gp = GaussianProcessRegressor(
            kernel=kernel_initial(),
            n_restarts_optimizer=10
        )
        self.estimation = None
        self.ucb = None
    
    def receive_prior_knowledge(self, X_train=None, y_train=None):
        if X_train is not None:
            assert X_train.shape[0] == y_train.shape[0], "Error: The number of elements in X_train and y_train must be equal."
            self.samples_X = np.concatenate((self.samples_X, X_train), axis=0)
            self.samples_Y = np.concatenate((self.samples_Y, y_train), axis=0)
            
    # step 1 (update neighbour postions + know responsible region + exchange samples)
    def voronoi_update(self, neighour_robot_index, robots_locations): # set responsible region and neighbours
        # 1) update neighbour postions
        self.neighbour = neighour_robot_index
        
        # 2) know responsible region
        # 采样精度是50*50，机器人实际的坐标是10*10
        
        scale_row = self.test_resolution[0]/self.field_row  
        scale_col = self.test_resolution[1]/self.field_col
        
        robots_locations = np.array(robots_locations)
        robots_locations_scaled = np.zeros_like(robots_locations)  # 创建一个与 robots 形状相同的全零数组
        robots_locations_scaled[:, 0] = robots_locations[:, 0] * scale_col  # 将第一列（X坐标）乘以10
        robots_locations_scaled[:, 1] = robots_locations[:, 1] * scale_row  # 将第二列（Y坐标）乘以20
        
        grid_points = np.array([(x, y) for x in range(self.test_resolution[0]) for y in range(self.test_resolution[1])])

        # 计算每个格点到所有机器人的距离
        distances = cdist(grid_points, robots_locations_scaled, metric='euclidean')

        # 找到每个格点最近的机器人
        closest_robot = np.argmin(distances, axis=1)

        # 标记距离 [0,0] 机器人最近的格点为 1
        self.responsible_region = np.zeros(self.test_resolution)
        
        for i, robot_index in enumerate(closest_robot):
            if robot_index == self.index:
                y, x = grid_points[i]
                self.responsible_region[x, y] = 1

        # 3) exchange samples
        exchange_dictionary_X = {}
        exchange_dictionary_y = {}
        
        closest_point_index_list = []
        
        sample_index = 0
        for sample in self.samples_X:
            if sample.tolist() not in self.sent_samples:
                distances = [distance.euclidean(sample, robo) for robo in robots_locations]
                # 找到距离最近的生成点的索引
                closest_point_index = np.argmin(distances)
                closest_point_index_list.append(closest_point_index)
                
                if closest_point_index != self.index:
                    if closest_point_index not in exchange_dictionary_X:
                        exchange_dictionary_X[closest_point_index] = []
                        exchange_dictionary_y[closest_point_index] = []  # 如果不存在，则初始化一个空列表
                    exchange_dictionary_X[closest_point_index].append(sample)
                    exchange_dictionary_y[closest_point_index].append(self.samples_Y[sample_index])
                    
                    self.sent_samples.append(sample.tolist())
            sample_index += 1

        return exchange_dictionary_X, exchange_dictionary_y
    
    def receive_samples(self, exchanged_samples_X, exchanged_samples_y):
        if exchanged_samples_X is not None:
            self.samples_X = np.concatenate((self.samples_X, exchanged_samples_X), axis=0)
            self.samples_Y = np.concatenate((self.samples_Y, exchanged_samples_y), axis=0)
    
    # step 2 calcualte GP
    def gp_regresssion(self, ucb_coeff=0.5): # the X_train, y_train is only for the prior knowledge
        # # 找到 samples_X 中重复元素的索引
        # 根据这些索引保留 samples_X 和 samples_Y 中的非重复元素
        _, unique_indices = np.unique(self.samples_X, axis=0, return_index=True)
        self.samples_X = self.samples_X[unique_indices]
        self.samples_Y = self.samples_Y[unique_indices]      
        self.gp.fit(self.samples_X, self.samples_Y)
        μ_test, σ_test = self.gp.predict(self.X_test, return_std=True)
        ucb = μ_test + ucb_coeff*σ_test
        
        self.estimation = μ_test.reshape(self.test_resolution)*self.responsible_region
        # plt.imshow(self.estimation, cmap='viridis')
        # plt.show()
        
        ucb = ucb.reshape(self.test_resolution)
        
        # plt.imshow(ucb, cmap='viridis')
        # cbar = plt.colorbar()
        # cbar.set_label('Value')
        # plt.show()
        
        self.phik = convert_phi2phik(self.Erg_ctrl.basis, ucb, self.grid)
        phi = convert_phik2phi(self.Erg_ctrl.basis, self.phik , self.grid)
       
        # plt.imshow(phi.reshape(50,50), cmap='viridis')
        # cbar = plt.colorbar()
        # cbar.set_label('Value')
        # plt.show()
        # self.Erg_ctrl.phik = convert_phi2phik(self.Erg_ctrl.basis, phi_vals, self.grid)
        return self.estimation, ucb
        # return μ_test, σ_test
    
    # step 3 exchange phik ck 
    def send_out_phik(self):
        if self.phik is not None:
            return self.phik
    
    def receive_phik_consensus(self, phik_pack):
        phik = [self.phik]
        if self.phik is not None:
            for neighbour_index in self.neighbour:
                phik.append(phik_pack[neighbour_index])
            phik_consensus = np.mean(phik, axis=0)
            self.Erg_ctrl.phik = 0.0025*phik_consensus
        phi = convert_phik2phi(self.Erg_ctrl.basis, phik_consensus , self.grid)
        # plt.imshow(phi.reshape(50,50), cmap='viridis')
        # cbar = plt.colorbar()
        # cbar.set_label('Value')
        # plt.show()   

        return phi.reshape(self.test_resolution)
        
    def send_out_ck(self):
        ck = self.Erg_ctrl.get_ck()
        if ck is not None:
            return ck 
    
    def receive_ck_consensus(self, ck_pack):
        my_ck = self.Erg_ctrl.get_ck()
        if my_ck is not None:
            cks = [my_ck]
            for neighbour_index in self.neighbour:
                cks.append(ck_pack[neighbour_index])
            ck_mean = np.mean(cks, axis=0)
            self.Erg_ctrl.receieve_consensus_ck(ck_mean)
        
    # step 4 move will ergodic control
    def get_nextpts(self, uniform=False, phi_vals=None):

        sample_steps = 1
        setpoints = []
        if uniform: 
            # setting the phik on the ergodic controller    
            phi_vals = np.ones(self.test_resolution)
            phi_vals /= np.sum(phi_vals)

            self.Erg_ctrl.phik = convert_phi2phik(self.Erg_ctrl.basis, phi_vals, self.grid)
        
        if phi_vals is not None:
            phi_vals = np.ones(self.test_resolution)
            phi_vals /= np.sum(phi_vals)

            self.Erg_ctrl.phik = convert_phi2phik(self.Erg_ctrl.basis, phi_vals, self.grid)
           
        i = 0
        print(np.mean(self.Erg_ctrl.phik))
        while (i < sample_steps):
            i += 1
            ctrl = self.Erg_ctrl(self.robot_state)
            self.robot_state = self.robot_dynamic.step(ctrl)
                  
            setpoints.append([ self.field_row*self.robot_state[0], self.field_col*self.robot_state[1] ])
 
        
        self.trajectory = self.trajectory + setpoints
        setpoints = np.array(setpoints)
        self.samples_X = np.concatenate((self.samples_X, setpoints), axis=0)
        self.samples_Y = np.concatenate((self.samples_Y, [f(setpoints)]), axis=0)
        
        return setpoints
    
    # Tools
    def get_trajectory(self):
        # return [self.field_row*self.robot_state[0], self.field_col*self.robot_state[1]]
        return self.trajectory
    
    # def _ck_link_callback(self, msg):
    #     received_robo_index = int(msg.name)
    #     if received_robo_index != self.index:
    #         if received_robo_index in self.neighbour: # is neighbour (add or update)
    #             # if received_robo_index in self._ck_dict: # update
    #             # else if received_robo_index not in self._ck_dict: # not in dict 
    #             self._ck_dict.update({received_robo_index : np.array(msg.ck)})
    #         elif received_robo_index in self._ck_dict: # not neighour any more 
    #             del self._ck_dict[received_robo_index]
    
    # def run(self):
    #     # rospy.init_node(self.agent_name)
    #     rate = rospy.Rate(10)
    #     while not rospy.is_shutdown():
    #         # self.step()
    #         rate.sleep() 
       # def _phik_link_callback(self, msg):
    #     return
    
    # def ck_concensus(self):
    #     my_ck = self.Erg_ctrl.get_ck()
    #     if len(self._ck_dict.keys()) >= 1:
    #         print("consensus!!!")
    #         cks = [my_ck]
    #         for key in self._ck_dict.keys():
    #             cks.append(self._ck_dict[key])
    #         ck_mean = np.mean(cks, axis=0)
    #         self.Erg_ctrl.set_ck(ck_mean)
    #         self._ck_msg.ck = ck_mean.copy()