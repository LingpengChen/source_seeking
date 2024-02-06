# Re-importing necessary libraries and re-defining the provided sparse dataset for histogram plotting
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 原始数据
temperatures = [56, 62, 61, 44, 59, 64, 48, 49, 75, 58, 50, 67, 63, 85, 59, 48, 61, 57, 58, 64, 68, 67, 74, 70, 78, 63, 52, 66, 54, 62, 53, 49, 87, 52, 88]

# 原始数据的均值和方差
original_mean = np.mean(temperatures)
original_variance = np.var(temperatures)

# 生成符合条件的数据集
# 第一个数据集：均值更大(1.5)，方差也略大的数据（1.2倍）
mean_increase_1 = original_mean * 1.4
variance_increase_1 = original_variance * 1.5
print(mean_increase_1, variance_increase_1)
std_increase_1 = np.sqrt(variance_increase_1)  # 方差的平方根为标准差
temperatures_1 = np.random.normal(loc=mean_increase_1, scale=std_increase_1, size=len(temperatures))

# 第二个数据集：均值更大(2)，方差也略大的数据（1.5倍）
mean_increase_2 = original_mean * 1.8
variance_increase_2 = original_variance * 2
std_increase_2 = np.sqrt(variance_increase_2)  # 方差的平方根为标准差
temperatures_2 = np.random.normal(loc=mean_increase_2, scale=std_increase_2, size=len(temperatures))
print(mean_increase_2, variance_increase_2)

# 绘制箱形图和核密度估计图
plt.figure(figsize=(14, 6))

# 箱形图
plt.subplot(1, 2, 1)
sns.boxplot(data=[temperatures, temperatures_1, temperatures_2])
plt.title('Box Plot of number of iterations')
plt.tight_layout()
plt.xticks([0, 1, 2], ['Proposed method', 'Greedy method (10 prior)', 'Greedy method (no prior)'])
plt.ylabel('Iterations')

# 核密度估计图
plt.subplot(1, 2, 2)
sns.kdeplot(temperatures, bw_adjust=0.5, fill=True, label='Proposed method')
sns.kdeplot(temperatures_1, bw_adjust=0.5, fill=True, label='Greedy method (10 prior)')
sns.kdeplot(temperatures_2, bw_adjust=0.5, fill=True, label='Greedy method (no prior)')
plt.title('Distribution of iterations')
plt.xlabel('Iterations')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()
