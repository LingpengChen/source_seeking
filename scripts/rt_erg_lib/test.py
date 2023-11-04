import numpy as np

# 假设 μ_estimation 和 f_X_test_reshaped 已经被定义并且都是 50x50 的矩阵
# μ_estimation = ...
# f_X_test_reshaped = ...

# 计算差异矩阵
μ_estimation = np.array([[1,1],[2,2]])
f_X_test_reshaped = np.array([[2,3],[2,2]])
import numpy as np

def calculate_wrmse(mu, mu_gt):
    # Element-wise absolute difference between prediction and ground truth
    
    # Global max and min across the ground truth matrix
    mu_gt_max = np.max(mu_gt)
    mu_gt_min = np.min(mu_gt)
    
    # Avoid division by zero in case max and min are the same
    if mu_gt_max == mu_gt_min:
        raise ValueError("Ground truth matrix has no range (max == min).")

    # Mean of the weighted squared differences
    mean_weighted_squared_diff = np.mean(np.abs(mu - mu_gt) * (mu_gt - mu_gt_min)  / (mu_gt_max - mu_gt_min))
    
    # Square root to get the WRMSE
    wrmse = np.sqrt(mean_weighted_squared_diff)
    
    return wrmse



wrmse_value = calculate_wrmse(μ_estimation, f_X_test_reshaped)
print("WRMSE:", wrmse_value)
