import numpy as np

def is_far_enough(new_point, existing_points, robot_locations, min_distance=2):
    """检查新点与现有点集和机器人初始位置是否保持足够距离"""
    # 检查与已生成点的距离
    if len(existing_points) > 0:
        for point in existing_points:
            if np.linalg.norm(new_point - point) < min_distance:
                return False
            
    # 检查与机器人初始位置的距离
    for robot_set in robot_locations:
        for robot_point in robot_set:
            if np.linalg.norm(new_point - np.array(robot_point)) < 1.5:
                return False
    return True

def generate_points(num_sets=10, points_per_set=5, min_distance=3, space_size=10):
    """生成符合条件的点集"""
    sets = []
    while len(sets) < num_sets:
        points = []
        iter_check = 0
        while len(points) < points_per_set:
            point = np.random.uniform(1.5, space_size - 1.5, 2)
            iter_check += 1
            if iter_check >= 1000:  # 防止无限循环
                break
            # 同时检查与现有点和机器人初始位置的距离
            if is_far_enough(point, points, ROBOT_INIT_LOCATIONS_case, min_distance):
                points.append(point)
        if len(points) == points_per_set:
            print(len(sets) )
            points = np.round(np.array(points) * 2) / 2
            sets.append(points)
    return sets

ROBOT_INIT_LOCATIONS_case = [[[1,2], [2, 2], [3,1]],
                            [[1,8], [2, 8], [3,9]]]

def format_points(sets_of_points):
    """
    Formats the sets of points into a string representation.

    :param sets_of_points: List of sets of points
    :return: List of formatted strings for each set of points
    """
    formatted_sets = []
    print("SOURCES = np.array([", end="")
    for points in sets_of_points:
        print( str(points.tolist()), ",")
    print("])")
    return 

# Parameters
num_sets = 10
num_points = 7
min_value = 1  # Min integer value
max_value = 9  # Max integer value (exclusive)

# Generate the points
sets_of_points = generate_points(num_sets=10, points_per_set=7, min_distance=3, space_size=10)
format_points(sets_of_points)

# np.array([[[5.0, 4.5], [7.5, 3.0], [5.0, 8.0], [2.0, 4.0], [8.0, 6.0], [2.5, 7.0], [4.0, 2.0]] ,
# [[2.5, 7.5], [5.0, 5.0], [7.0, 3.0], [2.0, 4.5], [8.0, 6.5], [4.0, 2.0], [5.5, 8.0]] ,
# [[4.5, 5.0], [7.5, 6.0], [5.5, 2.0], [4.5, 8.0], [2.0, 6.5], [2.0, 3.0], [8.0, 3.0]] ,
# [[5.5, 5.0], [2.0, 3.5], [4.5, 8.0], [8.0, 6.5], [4.5, 2.0], [8.0, 3.0], [2.0, 6.5]] ,
# [[2.0, 3.0], [5.5, 5.0], [4.5, 8.0], [5.5, 2.0], [8.0, 3.5], [8.0, 7.0], [2.0, 6.0]] ,
# [[3.5, 7.5], [3.5, 2.0], [8.0, 4.5], [6.5, 8.0], [5.0, 4.5], [6.5, 2.0], [2.0, 5.0]] ,
# [[5.5, 5.0], [2.0, 3.0], [4.5, 8.0], [5.0, 2.0], [2.0, 6.5], [8.0, 7.0], [8.0, 3.0]] ,
# [[2.0, 3.5], [7.5, 3.5], [4.5, 8.0], [8.0, 7.0], [5.0, 2.0], [4.5, 5.0], [2.0, 7.0]] ,
# [[5.5, 5.0], [5.5, 8.0], [2.5, 6.0], [4.5, 2.0], [8.0, 3.0], [8.0, 6.5], [2.0, 3.0]] ,
# [[5.5, 5.0], [4.5, 2.0], [5.0, 8.0], [2.0, 7.0], [2.0, 4.0], [8.0, 6.5], [8.0, 3.0]] ,
# ])