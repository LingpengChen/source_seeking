## Define the source field
from scipy.stats import multivariate_normal
import numpy as np
from matplotlib import pyplot as plt

DEBUG = True
FIELD_SIZE = [10, 10]

CAM_FOV = 0.4 # to distinguish whether this is one source or two source
SRC_MUT_D_THRESHOLD = 1

LCB_THRESHOLD = 0.15  
FIELD_SIZE_X = 10
FIELD_SIZE_Y = 10

USE_BO = True
BO_RADIUS = 0.5

CTR_MAG_DETERMIN_STUCK = 0.15
STUCK_PTS_THRESHOLD = 0.1 # smaller than this threshold are considered as the same stuck pt
# [[5,5], [9,9],[9,5], [5,9], [7,7]],
SOURCES_case = np.array([
[[2.0, 5.5], [6.5, 8.0], [4.5, 4.0], [8.0, 2.0], [3.5, 8.0], [2.0, 2.0], [8.0, 5.5]] ,
[[3.0, 7.5], [8.0, 6.5], [8.0, 3.0], [5.0, 2.0], [2.0, 5.0], [5.0, 5.0], [2.0, 2.0]] ,
[[5.0, 8.0], [2.0, 7.0], [7.5, 5.0], [5.0, 3.0], [2.0, 3.0], [8.0, 8.0], [8.0, 2.0]] ,
[[8.0, 5.0], [2.0, 6.5], [2.0, 2.0], [5.0, 4.5], [8.0, 2.0], [5.0, 7.5], [8.0, 8.0]] ,
[[5.5, 6.5], [6.5, 2.0], [2.5, 4.5], [2.5, 8.0], [8.0, 5.0], [8.0, 8.0], [3.5, 2.0]] ,
[[4.0, 8.0], [4.5, 3.5], [8.0, 4.5], [8.0, 8.0], [2.0, 2.0], [2.0, 5.5], [7.0, 2.0]] ,
[[6.0, 7.0], [5.0, 3.5], [2.5, 8.0], [8.0, 5.0], [2.0, 5.0], [7.5, 2.0], [2.5, 2.0]] ,
[[2.0, 5.0], [8.0, 6.5], [8.0, 3.5], [2.5, 8.0], [5.0, 2.0], [5.5, 5.0], [2.0, 2.0]] ,
[[5.0, 4.0], [7.5, 2.0], [6.0, 7.5], [2.0, 5.5], [2.0, 2.5], [8.0, 5.0], [3.0, 8.0]] ,
[[8.0, 3.0], [5.0, 7.5], [4.0, 4.5], [8.0, 8.0], [2.0, 7.5], [2.0, 2.0], [5.0, 2.0]] ,
[[5.5, 6.0], [2.0, 5.0], [5.5, 2.5], [8.0, 7.5], [3.5, 8.0], [2.5, 2.0], [8.0, 4.0]] ,
[[5.0, 2.5], [4.5, 5.5], [7.5, 5.5], [2.0, 7.5], [2.0, 3.5], [8.0, 2.5], [6.0, 8.0]] ,
])
ROBOT_INIT_LOCATIONS_case = [[[1,2], [2, 2], [3,1]],
                            [[1,8], [2, 8], [3,9]],
                            [[9,8], [8, 8], [7,9]],
                            [[9,2], [8, 2], [7,1]]]
def check_distance(sources, robot_locs):
    # Convert robot locations to numpy array for easier calculation
    robot_locs = np.array(robot_locs)
    
    # For each configuration of sources
    for sources_config in sources:
        # For each robot configuration
        for robots in robot_locs:
            # For each source in current configuration
            for source in sources_config:
                # For each robot in current configuration 
                for robot in robots:
                    # Calculate Euclidean distance
                    distance = np.sqrt((source[0] - robot[0])**2 + (source[1] - robot[1])**2)
                    if distance < 1:
                        print(f"Warning: Found distance < 1")
                        print(f"Source position: {source}")
                        print(f"Robot position: {robot}")
                        print(f"Distance: {distance}")
                        return False
    
    print("All distances are >= 1")
    return True

# Run the check
result = check_distance(SOURCES_case, ROBOT_INIT_LOCATIONS_case)