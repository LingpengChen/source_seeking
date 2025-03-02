from controller.double_integrator import DoubleIntegrator
from controller.ergodic_control import RTErgodicControl
from controller.mpc import MPCController
from scipy.spatial.distance import cdist
from scipy.spatial import distance

from controller.utils import convert_phi2phik, convert_phik2phi
import numpy as np

from environment.environment_and_measurement_7 import Environment, DEBUG
from environment.environment_and_measurement_7 import CAM_FOV, SRC_MUT_D_THRESHOLD, LCB_THRESHOLD, STUCK_PTS_THRESHOLD 

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from scipy.ndimage import maximum_filter

from utils.gpu_based_gp import TorchGPModel

def find_peak(matrix, strict=True):
    
    # 使用最大滤波器找到每个位置的局部最大值
    local_max = maximum_filter(matrix, size=3) == matrix # [[F F][T F]]
    
    # 获取局部最大值的坐标
    local_maxima_coords = np.argwhere(local_max)
    
    if strict:
        # 过滤出严格大于其周围邻居的局部最大值
        strict_local_maxima = []
        for i, j in local_maxima_coords:
            if i > 0 and j > 0 and i < matrix.shape[0] - 1 and j < matrix.shape[1] - 1:
                neighbors = [matrix[i-1, j-1], matrix[i-1, j], matrix[i-1, j+1],
                            matrix[i, j-1],                 matrix[i, j+1],
                            matrix[i+1, j-1], matrix[i+1, j], matrix[i+1, j+1]]
                if all(matrix[i, j] > neighbor for neighbor in neighbors):
                    strict_local_maxima.append([i,j])
        if (len(strict_local_maxima)):
            return np.array(strict_local_maxima)[:, [1, 0]]
        else:
            return strict_local_maxima

    else:
        return local_maxima_coords[:, [1, 0]]

def kernel_initial(
            σf_initial=1.0,         # covariance amplitude
            ell_initial=1.0,        # length scale
            σn_initial=0.1          # noise level
        ):
            return σf_initial**2 * RBF(length_scale=ell_initial, length_scale_bounds=(0.5, 2)) + WhiteKernel(noise_level=σn_initial)

class Robot(object): # python (x,y) therefore col index first, row next
    # robot state + controller initialization!
    def __init__(self, start_position, index, environment: Environment, test_resolution = [50,50]):
        
        ## robot index
        self.index = index
        self.agent_name = str(index)
        
        ## field information
        # test_resolution
        self.test_resolution = test_resolution # [50, 50]
        # real size
        self.environment = environment
        self.field_size = self.environment.field_size
        
        # for ES
        grid_2_r_w = np.meshgrid(np.linspace(0, 1, int(test_resolution[0])), np.linspace(0, 1, int(test_resolution[1])))
        self.grid = np.c_[grid_2_r_w[0].ravel(), grid_2_r_w[1].ravel()] # (2500,2)

        # for gp
        X_test_x = np.linspace(0, self.field_size[0], test_resolution[0])
        X_test_y = np.linspace(0, self.field_size[1], test_resolution[1])
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
        self.Mpc_ctrl    = MPCController(DoubleIntegrator(), horizon=10, Q=1, R=0.01)
        # samples 
        self.samples_X = np.array([start_position])
        self.samples_Y = np.array(self.environment.sampling([start_position]))
        
        self.trajectory = [start_position]
        
        ## ROBOT neigibours
        self.neighbour = None  
        self.responsible_region = np.zeros(self.test_resolution)
        if DEBUG:
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
        self.peaks_LCB = None
        self.LCB_THRESHOLD = LCB_THRESHOLD
        self.iteration = 0
        
        self.visited_peaks_cord = []
        self.stuck_points = []
        self.stuck_times = 0
        ## MI
        self.gamma_gp = 0
        
        self.target = None
    
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
        
        self.responsible_region = np.zeros(self.test_resolution)
        mask = (closest_robot == self.index)
        self.responsible_region[grid_points[mask, 1], grid_points[mask, 0]] = 1      

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
    def gp_learn_and_get_acquisition(self, ucb_coeff=2): # the X_train, y_train is only for the prior knowledge
        # # 找到 samples_X 中重复元素的索引
        # 根据这些索引保留 samples_X 和 samples_Y 中的非重复元素
        _, unique_indices = np.unique(self.samples_X, axis=0, return_index=True)
        self.samples_X = self.samples_X[unique_indices]
        self.samples_Y = self.samples_Y[unique_indices]      
        self.gp.fit(self.samples_X, self.samples_Y)
        # Bayesian inference
        μ_test, σ_test = self.gp.predict(self.X_test, return_std=True)
        
        self.estimation = μ_test.reshape(self.test_resolution)
        self.variance = σ_test.reshape(self.test_resolution)
        
        # update gamma based on variance of xt from the last iteration, i.e., gama_t-1
        if len(self.trajectory) > 1:
            _, variance = self.gp.predict([self.trajectory[-1]], return_std=True)
            self.gamma_gp += variance
        
        # definition of phi_t(x) for all x
        mutual_info_dist = μ_test + ucb_coeff * (np.sqrt(σ_test + self.gamma_gp) - np.sqrt(self.gamma_gp))
        self.gp_mi = mutual_info_dist.reshape(self.test_resolution)
        
         # calculate phik based on MI
        self.phik = convert_phi2phik(self.Erg_ctrl.basis, self.gp_mi, self.grid)
        phi = convert_phik2phi(self.Erg_ctrl.basis, self.phik , self.grid)
        
        return self.estimation*self.responsible_region, self.gp_mi*self.responsible_region
    
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
    def get_nextpts(self, phi_vals=None, control_mode="NORMAL"):
        self.iteration += 1
        ctrl = None
        self.target = None
        active_sensing = True   
        if control_mode == "ES_UNIFORM":
            # setting the phik on the ergodic controller    
            phi_vals = np.ones(self.test_resolution)
            phi_vals /= np.sum(phi_vals)
            self.Erg_ctrl.phik = convert_phi2phik(self.Erg_ctrl.basis, phi_vals, self.grid)
            ctrl = self.Erg_ctrl(self.robot_state)
           
        elif control_mode == "NORMAL":
            if phi_vals is not None:
                phi_vals /= np.sum(phi_vals)
                self.Erg_ctrl.phik = convert_phi2phik(self.Erg_ctrl.basis, phi_vals, self.grid)

            # determine whether to active sensing or source seeking
            
            # remove the peak that has already been visited
            indices_to_remove = []
            for i, peak in enumerate(self.peaks_cord):
                for visited_peak in (self.visited_peaks_cord):
                    if np.linalg.norm(np.array(peak) - np.array(visited_peak)) < SRC_MUT_D_THRESHOLD:
                        indices_to_remove.append(i)
                        break
            
            # remove the peak that has already been visited
            for i, peak in enumerate(self.peaks_cord):
                for stuck_pts in (self.stuck_points):
                    if np.linalg.norm(np.array(peak) - np.array(stuck_pts)) < STUCK_PTS_THRESHOLD:
                        indices_to_remove.append(i)
                        break
            
            # all potential peaks that haven't been visited
            peaks_cord = [peak for i, peak in enumerate(self.peaks_cord) if i not in indices_to_remove]
            peaks_LCB = [lcb for i, lcb in enumerate(self.peaks_LCB) if i not in indices_to_remove]
            
            # determine whether there is a peak of good confidence  
            lcb_value = self.LCB_THRESHOLD - 0.002*self.iteration
            if peaks_LCB and np.max(peaks_LCB) > lcb_value:   
                index = np.argmax(peaks_LCB)
                distance = cdist([self.trajectory[-1]], [peaks_cord[index]])[0][0]
                self.target = peaks_cord[index]
                active_sensing = False
            
            # calculate control command
            if active_sensing:
                ctrl = self.Erg_ctrl(self.robot_state)
                if DEBUG:
                    print("ES: ", np.linalg.norm(ctrl))
                
            else: # source seeking
                self.Erg_ctrl.update_trajectory(self.robot_state) # update ck
                ctrl_target = np.array(self.target) / self.field_size[0]
                ctrl = self.Mpc_ctrl(self.robot_state, ctrl_target)
                if DEBUG:
                    print("mpc: ", np.linalg.norm(ctrl), "with target: ", self.target)

        # move based on ctrl command
        self.robot_state = self.robot_dynamic.step(ctrl)         
        setpoint = [[ self.field_size[0]*self.robot_state[0], self.field_size[1]*self.robot_state[1] ]]

        stepsize = np.linalg.norm( np.array(setpoint)-np.array(self.trajectory[-1]) )
        if DEBUG:
            print("stepsize = ", stepsize)
        if (not active_sensing) and stepsize < 0.1: # get stuck at none sources area 
        # if (not active_sensing) and stepsize < 0.1 and np.linalg.norm(ctrl) < CTR_MAG_DETERMIN_STUCK: # get stuck at none sources area 
            if (self.stuck_times >= 2):
                self.stuck_points.append(setpoint)
                if DEBUG:
                    print("The stuck point is: ", self.stuck_points)
                self.stuck_times = 0
            else:
                if DEBUG:
                    print("SLOW! with step size", stepsize)
                self.stuck_times += 1
        else:
            self.stuck_times = 0
            
        self.trajectory = self.trajectory + setpoint
        setpoint = np.array(setpoint)
        
        # determine whether the source is found
        # some special sensors here (maybe camera here to determine whether a source is found)
        source_cord = self.environment.find_source(setpoint)
        if (source_cord is not None) and (self.target is not None):
            if DEBUG:
                print("source", source_cord, "is found by Robot ", self.index, "!", "The target is ", self.target)
            self.visited_peaks_cord.append(list(source_cord))

        # take samples and add to local dataset
        self.samples_X = np.concatenate((self.samples_X, setpoint), axis=0)
        self.samples_Y = np.concatenate((self.samples_Y, self.environment.sampling(setpoint)), axis=0)
        
        return setpoint
    
    # Tools
    def get_trajectory(self):
        # return [self.field_size[0]*self.robot_state[0], self.field_size[1]*self.robot_state[1]]
        return self.trajectory
    
    def estimate_source(self, lcb_coeff=2):
        peaks = np.array(find_peak(self.estimation)) # [col_mat, ...]

        real_res_ratio = self.field_size[0]/self.test_resolution[0] # 10m/50 = 0.2m
        increased_resolution_ratio = self.test_resolution[0]/2 # 25
        real_res_ratio_new = real_res_ratio/increased_resolution_ratio
        X_test = self.X_test/increased_resolution_ratio # increase resolution of sampling point [[0,0.2]] => [[0,0.2/ratio]]
        peaks_cord = [] # this is all peaks without filtering (over the whole field)
        for peak in peaks:
            x, y =  real_res_ratio * (peak[0]-1),  real_res_ratio * (peak[1]-1) # real coordinate (start from upper left)
            X_test_copy = np.copy(X_test)
            X_test_copy[:, 0] += x
            X_test_copy[:, 1] += y  
            μ_test, σ_test = self.gp.predict(X_test_copy, return_std=True)
            new_peaks = np.array(find_peak(μ_test.reshape(self.test_resolution), strict=False)[0])
            
            peak_x, peak_y = x + real_res_ratio_new*new_peaks[0], y + real_res_ratio_new*new_peaks[0]
            peaks_cord.append([peak_x, peak_y])

        self.peaks_cord = [] # filter out the peaks out of the voronoi cell
        self.peaks_LCB = []
        if len(peaks_cord):
            μ, σ = self.gp.predict(peaks_cord, return_std=True)
            own_loc = self.trajectory[-1]
            neighbour_loc = self.neighbour_loc
            bots_loc = [own_loc] + neighbour_loc
            distances = cdist(np.array(peaks_cord), bots_loc, metric='euclidean')
            closest_robot_index = np.argmin(distances, axis=1)
            
            i=0
            for peak in peaks_cord:
                if closest_robot_index[i] == 0: 
                    LCB = μ[i] - lcb_coeff*σ[i]
                    self.peaks_LCB.append(LCB)
                    self.peaks_cord.append(peak)
                i+=1
            
        return self.peaks_cord, self.peaks_LCB
    
    