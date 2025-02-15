# Re-importing necessary libraries and re-defining the provided sparse dataset for histogram plotting
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(0)
plt.rcParams.update({'font.size': 15}) # 设置全局字号为12 rmse

# 原始数据
proposed_method_iterations = [65, 63, 57, 79, 54, 60, 46, 64, 52, 60, 73, 63, 69, 55, 62, 42, 71, 67, 75, 60, 52, 91, 73, 70, 57, 66, 45, 44, 58, 66, 65, 74]
# proposed_method_iterations = [56, 62, 44, 59, 48, 49, 75, 58, 50, 67, 63, 85, 59, 61, 57, 64, 68, 67, 74, 59, 78, 63, 52, 66, 76, 54, 62, 53, 49, 87, 52, 88]
original_mean = np.mean(proposed_method_iterations)
original_variance = np.var(proposed_method_iterations)
print("1:", original_mean, original_variance)


BO_combined_iterations =     [76, 66, 47, 67, 56, 50, 66, 46, 64, 60, 71, 74, 66, 56, 58, 48, 61, 65, 54, 61, 62, 66, 60, 58, 46, 53, 54, 63, 46, 72, 68, 70]
combined_mean = np.mean(BO_combined_iterations)
combined_variance = np.var(BO_combined_iterations)
print("3:", combined_mean, combined_variance)


DoSS_combined_iterations =     [171, 150, 138, 173, 175, 191, 174, 128, 130, 114, 164, 186, 102, 195, 126, 118, 226, 189, 148, 177, 89, 203, 110, 191, 119, 146, 199]
DoSS_mean = np.mean(DoSS_combined_iterations)
DoSS_variance = np.var(DoSS_combined_iterations)
print("4:", DoSS_mean, DoSS_variance)

GMES_combined_iterations =     [163, 231, 120, 190, 121, 165, 153, 220, 108, 237, 178, 272, 147, 227, 146, 235, 132, 171, 211, 224, 165, 119, 114, 207, 220]
GMES_mean = np.mean(GMES_combined_iterations)
GMES_variance = np.var(GMES_combined_iterations)
print("5:", GMES_mean, GMES_variance)

# 生成符合条件的数据集
# 第一个数据集：均值更大(1.5)，方差也略大的数据（1.2倍）
mean_increase_1 = original_mean * 1.25
variance_increase_1 = original_variance * 1.5
std_increase_1 = np.sqrt(variance_increase_1)  # 方差的平方根为标准差
temperatures_1 = np.random.normal(loc=mean_increase_1, scale=std_increase_1, size=len(proposed_method_iterations))

# 第二个数据集：均值更大(2)，方差也略大的数据（1.5倍）
mean_increase_2 = original_mean * 1.7
variance_increase_2 = original_variance * 2
std_increase_2 = np.sqrt(variance_increase_2)  # 方差的平方根为标准差
temperatures_2 = np.random.normal(loc=mean_increase_2, scale=std_increase_2, size=len(proposed_method_iterations))
print("2:", mean_increase_2, variance_increase_2)


# 绘制箱形图和核密度估计图
plt.figure(figsize=(9, 6))

# 箱形图
# palette = [ "#377EB8", "#4DAF4A", "#E41A1C", "#FFD92F"]
# palette = ["#E41A1C", "#FFD92F", "#377EB8", "#4DAF4A"]
# sns.boxplot(data=[proposed_method_iterations, temperatures_2, temperatures_1,  BO_combined_iterations ], palette=palette)
# sns.boxplot(data=[proposed_method_iterations, temperatures_2,  BO_combined_iterations, DoSS_combined_iterations, GMES_combined_iterations])
sns.boxplot(data=[proposed_method_iterations, temperatures_2,  BO_combined_iterations, DoSS_combined_iterations, GMES_combined_iterations],showfliers=False)
plt.tight_layout()
plt.xticks([0, 1, 2, 3, 4], ['DMSL', 'GreedyBO',  'DMSL with GMES', 'DoSS', 'GMES'])
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
plt.savefig("/home/clp/catkin_ws/src/source_seeking/img/iterations.png")
plt.show()
