import numpy as np


# utils for experiment
def calculate_wrmse(mu, mu_gt):
    # Element-wise absolute difference between prediction and ground truth
    
    # Global max and min across the ground truth matrix
    mu_gt_max = np.max(mu_gt)
    mu_gt_min = np.min(mu_gt)
    
    # Avoid division by zero in case max and min are the same
    if mu_gt_max == mu_gt_min:
        raise ValueError("Ground truth matrix has no range (max == min).")

    # Mean of the weighted squared differences
    mean_weighted_squared_diff = np.mean(((mu - mu_gt) ** 2) * (mu_gt - mu_gt_min)  / (mu_gt_max - mu_gt_min))
    # mean_weighted_squared_diff = np.mean(np.abs(mu - mu_gt) * (mu_gt - mu_gt_min)  / (mu_gt_max - mu_gt_min))
    
    # Square root to get the WRMSE
    wrmse = np.sqrt(mean_weighted_squared_diff)
    
    return wrmse

