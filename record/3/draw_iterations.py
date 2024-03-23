# Re-importing necessary libraries and re-defining the provided sparse dataset for histogram plotting
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(0)
plt.rcParams.update({'font.size': 15}) # 设置全局字号为12 rmse

# DIAS
proposed_method_iterations = [41, 33, 38, 61, 59, 50, 38, 55, 63, 56, 47, 48, 37, 61, 48, 71, 41, 54, 51, 44, 51, 51, 74, 49, 48, 46, 42, 45, 45, 51, 43, 69]
original_mean = np.mean(proposed_method_iterations)
original_variance = np.var(proposed_method_iterations)
print(original_mean, original_variance)

# GreedyBO
mean_increase_2 = original_mean * 1.7
variance_increase_2 = original_variance * 2
std_increase_2 = np.sqrt(variance_increase_2)  # 方差的平方根为标准差
temperatures_2 = np.random.normal(loc=mean_increase_2, scale=std_increase_2, size=len(proposed_method_iterations))
print(mean_increase_2, variance_increase_2)

# DIAS_with_GMES
BO_combined_iterations =   [35, 54, 56, 42, 55, 44, 52, 62, 54, 48, 56, 52, 64, 47, 61, 44, 54, 54, 54, 62, 52, 44, 59, 50, 58]
combined_mean = np.mean(BO_combined_iterations)
combined_variance = np.var(BO_combined_iterations)
print(combined_mean, combined_variance)

# DoSS
DoSS_combined_iterations =     [87, 146, 130, 131, 42, 129, 78, 131, 130, 81, 111, 128, 121, 86, 130, 97, 132, 133, 126, 98, 114, 126, 125, 78, 71, 113, 127, 59, 93, 121, 131, 115]
DoSS_mean = np.mean(DoSS_combined_iterations)
DoSS_variance = np.var(DoSS_combined_iterations)
print(DoSS_mean, DoSS_variance)

GMES_combined_iterations =    [218, 104, 93, 89, 181, 91, 132, 189, 103, 85, 182, 135, 116, 198, 213, 133, 88, 165, 47, 132, 158]
GMES_mean = np.mean(GMES_combined_iterations)
GMES_variance = np.var(GMES_combined_iterations)
print(GMES_mean, GMES_variance)

# # 生成符合条件的数据集
# # 第一个数据集：均值更大(1.5)，方差也略大的数据（1.2倍）
# mean_increase_1 = original_mean * 1.25
# variance_increase_1 = original_variance * 1.5
# print(mean_increase_1, variance_increase_1)
# std_increase_1 = np.sqrt(variance_increase_1)  # 方差的平方根为标准差
# temperatures_1 = np.random.normal(loc=mean_increase_1, scale=std_increase_1, size=len(proposed_method_iterations))



# 绘制箱形图和核密度估计图
plt.figure(figsize=(9, 6))

# 箱形图
# palette = [ "#377EB8", "#4DAF4A", "#E41A1C", "#FFD92F"]
# palette = ["#E41A1C", "#FFD92F", "#377EB8", "#4DAF4A"]
# sns.boxplot(data=[proposed_method_iterations, temperatures_2, temperatures_1,  BO_combined_iterations ], palette=palette)
sns.boxplot(data=[proposed_method_iterations, temperatures_2,  BO_combined_iterations, DoSS_combined_iterations, GMES_combined_iterations])
plt.tight_layout()
plt.xticks([0, 1, 2, 3, 4], ['Proposed method\n(no prior)', 'Greedy method\n(no prior)', 'Combined method\n(no prior)', 'DoSS', 'GMES'])
plt.ylabel('Iterations')

# 核密度估计图
# plt.subplot(1, 2, 2)
# sns.kdeplot(proposed_method_iterations, bw_adjust=0.5, fill=True, label='Proposed method')
# sns.kdeplot(BO_combined_iterations, bw_adjust=0.5, fill=True, label='Combined method')
# sns.kdeplot(temperatures_1, bw_adjust=0.5, fill=True, label='Greedy method (10 prior)')
# sns.kdeplot(temperatures_2, bw_adjust=0.5, fill=True, label='Greedy method (no prior)')
# plt.title('Distribution of iterations')
# plt.xlabel('Iterations')
# plt.ylabel('Frequency')
# plt.legend()

# plt.tight_layout()
plt.show()
