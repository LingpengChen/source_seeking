import sys
# sys.path.append('./rt_erg_lib/')
from double_integrator import DoubleIntegrator
from ergodic_control import RTErgodicControl
from mpc import MPCController
from target_dist import TargetDist
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi, distance
import matplotlib.pyplot as plt

from utils import convert_phi2phik, convert_ck2dist, convert_traj2ck, convert_phik2phi, find_peak
import numpy as np
from scipy.io import loadmat

import rospy
from grid_map_msgs.msg import GridMap
from source_seeking.msg import Ck
from environment_and_measurement import sampling, find_source, FOUND_SOURCE_THRESHOLD, LCB_THRESHOLD # sampling function is just f with noise

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
        self.field_size = field_size # [10, 10]
        
        # for ES
        grid_2_r_w = np.meshgrid(np.linspace(0, 1, int(test_resolution[0])), np.linspace(0, 1, int(test_resolution[1])))
        self.grid = np.c_[grid_2_r_w[0].ravel(), grid_2_r_w[1].ravel()] # (2500,2)

        # for gp
        X_test_x = np.linspace(0, field_size[0], test_resolution[0])
        X_test_y = np.linspace(0, field_size[1], test_resolution[1])
        X_test_xx, X_test_yy = np.meshgrid(X_test_x, X_test_y)
        self.X_test = np.vstack(np.dstack((X_test_xx, X_test_yy))) # (2500,2)
        
        ## ROBOT information
        # robot dynamics
        self.robot_dynamic = DoubleIntegrator() # robot controller system
        self.robot_state     = np.zeros(self.robot_dynamic.observation_space.shape[0])
        # robot initial location
        self.robot_state[:2] = np.array([start_position[0]/self.field_size[0], start_position[1]/self.field_size[1]])
        self.robot_dynamic.reset(self.robot_state)
        self.Erg_ctrl    = RTErgodicControl(DoubleIntegrator(), weights=0.01, horizon=15, num_basis=10, batch_size=-1)
        self.Mpc_ctrl    = MPCController(DoubleIntegrator(), horizon=20, Q=1, R=0.001)
        # samples 
        self.samples_X = np.array([start_position])
        self.samples_Y = np.array(sampling([start_position]))
        
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
        self.variance = None
        self.peaks_cord = None
        self.peaks_UCB = None
        self.ucb = None
        self.visited_peaks_cord = []
        self.stuck_points = []
        ## 
    
    def receive_prior_knowledge(self, X_train=None, y_train=None):
        if X_train is not None:
            assert X_train.shape[0] == y_train.shape[0], "Error: The number of elements in X_train and y_train must be equal."
            self.samples_X = np.concatenate((self.samples_X, X_train), axis=0)
            self.samples_Y = np.concatenate((self.samples_Y, y_train), axis=0)
            
    # step 1 (update neighbour postions + know responsible region + exchange samples)
    def voronoi_update(self, neighour_robot_index, robots_locations): # set responsible region and neighbours
        # 1) update neighbour postions
        self.neighbour = neighour_robot_index
        self.neighbour_loc = [robots_locations[i] for i in self.neighbour]
        # 2) know responsible region
        # 采样精度是50*50，机器人实际的坐标是10*10
        
        scale_row = self.test_resolution[0]/self.field_size[0]
        scale_col = self.test_resolution[1]/self.field_size[1]
        
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
    def gp_regresssion(self, ucb_coeff=2): # the X_train, y_train is only for the prior knowledge
        # # 找到 samples_X 中重复元素的索引
        # 根据这些索引保留 samples_X 和 samples_Y 中的非重复元素
        _, unique_indices = np.unique(self.samples_X, axis=0, return_index=True)
        self.samples_X = self.samples_X[unique_indices]
        self.samples_Y = self.samples_Y[unique_indices]      
        self.gp.fit(self.samples_X, self.samples_Y)
        μ_test, σ_test = self.gp.predict(self.X_test, return_std=True)
        
        
        ucb = μ_test + ucb_coeff*σ_test
        self.estimation = μ_test.reshape(self.test_resolution)
        self.variance = σ_test.reshape(self.test_resolution)

        ucb = ucb.reshape(self.test_resolution)
        
        # calculate phik based on ucb
        self.phik = convert_phi2phik(self.Erg_ctrl.basis, ucb, self.grid)
        phi = convert_phik2phi(self.Erg_ctrl.basis, self.phik , self.grid)

        self.ucb = ucb 
        self.estimate_source(ucb_coeff)
        return self.estimation*self.responsible_region, self.ucb*self.responsible_region
    
    # step 3 exchange phik ck visted source
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
    
    def send_out_source_cord(self):
        return self.visited_peaks_cord
    
    def receive_source_cord(self, peaks):
        self.visited_peaks_cord = peaks
        
    # step 4 move will ergodic control
    def get_nextpts(self, phi_vals=None, control_mode="UCB_greedy"):
        ctrl = None
        target = None    
           
        if control_mode == "UCB_greedy":
            # remove the peak that has already been visited
            indices_to_remove = []
            for i, peak in enumerate(self.peaks_cord):
                for visited_peak in (self.visited_peaks_cord+self.stuck_points):
                    if np.linalg.norm(np.array(peak) - np.array(visited_peak)) < 2*FOUND_SOURCE_THRESHOLD:
                        indices_to_remove.append(i)
                        break
            peaks_cord = [peak for i, peak in enumerate(self.peaks_cord) if i not in indices_to_remove]
            peaks_UCB = [ucb_value for i, ucb_value in enumerate(self.peaks_UCB) if i not in indices_to_remove]
            
            if len(peaks_cord) > 0:
                index = np.argmax(peaks_UCB)
                target = peaks_cord[index]
                ctrl_target = np.array(target) / self.field_size[0]
                ctrl = self.Mpc_ctrl(self.robot_state, ctrl_target)
                
                if np.linalg.norm(ctrl) < 0.1: # get stuck at none sources area 
                    self.stuck_points.append(target)
                    # print("For robot", i, "All the peak within the cell has been visited, so let's get to the maximum places ")
            
            else: # the peak within the cell has been visited
                # The maximum UCB point within the region
                selfucb = self.ucb * self.responsible_region
                ucb_value = np.max(selfucb)
                max_index_1d = np.argmax(selfucb.T)
                max_index_2d = np.unravel_index(max_index_1d, selfucb.shape)
                target = np.array(max_index_2d) * (self.field_size[0] / self.test_resolution[0])
                ctrl_target = np.array(target) / self.field_size[0]
                ctrl = self.Mpc_ctrl(self.robot_state, ctrl_target)
                
        # elif control_mode == "UCB_greedy":
        #     # remove the peak that has already been visited
        #     max_index_1d = np.argmax(self.ucb.T)
        #     # 使用 unravel_index 将一维索引转换为二维索引
        #     max_index_2d = np.unravel_index(max_index_1d, self.ucb.shape)
        #     ctrl_target = np.array(max_index_2d) / self.test_resolution[0]
        #     target = ctrl_target * self.field_size[0]
        #     ctrl = self.Mpc_ctrl(self.robot_state, ctrl_target)
        self.robot_state = self.robot_dynamic.step(ctrl)         
        setpoint = [[ self.field_size[0]*self.robot_state[0], self.field_size[1]*self.robot_state[1] ]]
       
        self.trajectory = self.trajectory + setpoint
        setpoint = np.array(setpoint)
        
        # determine whether the source is found
        source_cord = find_source(setpoint, FOUND_SOURCE_THRESHOLD)
        if source_cord is not None:
            print("source", source_cord, "is found by Robot ", self.index, "!", "The target is ", target)
            self.visited_peaks_cord.append(list(source_cord))

        self.samples_X = np.concatenate((self.samples_X, setpoint), axis=0)
        self.samples_Y = np.concatenate((self.samples_Y, sampling(setpoint)), axis=0)
        
        return setpoint, target
    
    # Tools
    def get_trajectory(self):
        # return [self.field_size[0]*self.robot_state[0], self.field_size[1]*self.robot_state[1]]
        return self.trajectory
    
    def estimate_source(self, ucb_coeff):
        peaks = find_peak(self.ucb, strict=False)
        peaks_cord = np.array(peaks) * (self.field_size[0]/self.test_resolution[0]) # [col_mat, ...]
            

        self.peaks_cord = [] # filter out the peaks out of the voronoi cell
        self.peaks_UCB = []
        
        
        μ, σ = self.gp.predict(peaks_cord, return_std=True)
        own_loc = self.trajectory[-1]
        neighbour_loc = self.neighbour_loc
        bots_loc = [own_loc] + neighbour_loc
        distances = cdist(np.array(peaks_cord), bots_loc, metric='euclidean')
        closest_robot_index = np.argmin(distances, axis=1)
        
        i=0
        for peak in peaks_cord: # find the peak within the Voronoi cell
            if closest_robot_index[i] == 0: 
                UCB = μ[i] + ucb_coeff*σ[i]
                self.peaks_UCB.append(UCB)
                self.peaks_cord.append(list(peak))
            i+=1

      
        
        return self.peaks_cord, self.peaks_UCB

    
    def get_estimated_source(self):
        return self.peaks_cord, self.peaks_UCB 
    