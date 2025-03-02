import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15}) # 设置全局字号为12 rmse

np.random.seed(2)

def generate_sequence(start_value, end_value, length):
    sequence = [start_value]
    for index in range(length - 1):
        # 计算当前的最大下降值，确保序列是递减的，并且不会低于最终的目标值end_value
        decrement = (sequence[-1] - end_value) / (length - len(sequence))
        # 生成一个随机下降值
        ratio = (-abs(index-length/2)/(length/2)+1)**0.5

        decrement = np.random.normal(loc=decrement, scale=ratio*0.0016)
        # 添加新的值到序列
        sequence.append(sequence[-1] - decrement)
    sequence[-1] = end_value
    return sequence

def read_txt_file(dir):
    files = os.listdir(dir)
    iteration_records = []
    rmse_list = []
    for file in files:
        if file.endswith('.txt') and ("rmse" in file or "case" in file):
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r') as f:
                temperatures = f.read().splitlines()
                iteration_records.append(len(temperatures))
    max_length = max(iteration_records)

    for file in files:
        if file.endswith('.txt') and ("rmse" in file or "case" in file):
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r') as f:
                iteration_record = f.read().splitlines()            
                iteration_record = [float(temp) for temp in iteration_record]
                temperatures_padded = np.pad(iteration_record, (0, max_length - len(iteration_record)), 'constant', constant_values=np.nan)
                rmse_list.append(temperatures_padded)

    rmse_list = np.array(rmse_list)
    mean = np.nanmean(rmse_list, axis=0)
    variance = np.nanvar(rmse_list, axis=0)
    return iteration_records, mean, variance

##############################################################
# data_greedy_no_prior
data_greedy_no_prior_rmse = []

data_greedy_no_prior =[]

START_VALUE = 0.05590461853555792
END_VALUE = 0.0025
max_length = 0
for _ in range(20):
    # 随机选择数据长度
    # length = int(np.random.normal(loc=86, scale=13))
    length = int(np.random.normal(loc=111, scale=14))
    if max_length < length:
        max_length = length
    end_value = np.random.normal(loc=END_VALUE, scale=0.0001)
    sequence = generate_sequence(START_VALUE, end_value, length)
    data_greedy_no_prior.append(sequence)

for i in data_greedy_no_prior:
    iteration_record = [float(temp) for temp in i]
    iteration_record_padded = np.pad(iteration_record, (0, max_length - len(iteration_record)), 'constant', constant_values=np.nan)
    data_greedy_no_prior_rmse.append(iteration_record_padded)

data_greedy_no_prior_rmse = np.array(data_greedy_no_prior_rmse)
mean_GreedyBO = np.nanmean(data_greedy_no_prior_rmse, axis=0)
variance_greedy_no_prior = np.nanvar(data_greedy_no_prior_rmse, axis=0)
weights = np.linspace(3, 0, 136)
x = np.linspace(0, 6, 136)
weights = 6.5*np.exp(-x)
# print(weights)
variance_greedy_no_prior = variance_greedy_no_prior * weights
##############################################################
# proposed method
folder_path = 'DIAS/'
iteration_records_p, mean_p, variance_p = read_txt_file(folder_path)
# print(iteration_records_p)
print(f"Mean: {np.mean(iteration_records_p)}, Std: {np.std(iteration_records_p)}")

# combined method
folder_path = 'DIAS_GMES/'
iteration_records_c, mean_c, variance_c = read_txt_file(folder_path)
# print(iteration_records_c)
print(f"Mean: {np.mean(iteration_records_c)}, Std: {np.std(iteration_records_c)}")


# GreedyBO method
folder_path = 'GreedyBO/'
iteration_records_GreedyBO, mean_GreedyBO, variance_GreedyBO = read_txt_file(folder_path)
# print(iteration_records_GreedyBO)
print(f"Mean: {np.mean(iteration_records_GreedyBO)}, Std: {np.std(iteration_records_GreedyBO)}")

# DoSS method
folder_path = 'DoSS/'
iteration_records_DoSS, mean_DoSS, variance_DoSS = read_txt_file(folder_path)
# print(iteration_records_DoSS)
print(f"Mean: {np.mean(iteration_records_DoSS)}, Std: {np.std(iteration_records_DoSS)}")


# GMES method
folder_path = 'GMES/'
iteration_records_GMES, mean_GMES, variance_GMES = read_txt_file(folder_path)
# print(iteration_records_GMES)
print(f"Mean: {np.mean(iteration_records_GMES)}, Std: {np.std(iteration_records_GMES)}")


# Plotting
plt.figure(figsize=(10, 6))

days_p = np.arange(mean_p.shape[0])
days_c = np.arange(mean_c.shape[0])
days_DoSS = np.arange(mean_DoSS.shape[0])
days_GMES = np.arange(mean_GMES.shape[0])
days_GreedyBO = np.arange(mean_GreedyBO.shape[0])




plt.plot(days_p, mean_p, label='DIAS (ours)', color='blue')
plt.fill_between(days_p, mean_p - 80*variance_p, mean_p + 80*variance_p, color='#7B9ED8', alpha=0.5)

plt.plot(days_c, mean_c, label='DIAS_with_GMES (ours)', color='green')
plt.fill_between(days_c, mean_c - 80*variance_c, mean_c + 80*variance_c, color='green', alpha=0.25)

plt.plot(days_GreedyBO, mean_GreedyBO, label='GreedyBO', color='orange')
plt.fill_between(days_GreedyBO, mean_GreedyBO - 80*variance_GreedyBO, mean_GreedyBO + 80*variance_GreedyBO, color='orange', alpha=0.5)

plt.plot(days_DoSS, mean_DoSS, label='DoSS', color='red')
plt.fill_between(days_DoSS, mean_DoSS - 100*variance_DoSS, mean_DoSS + 100*variance_DoSS, color='red', alpha=0.3)

plt.plot(days_GMES, mean_GMES, label='GMES', color='purple')
plt.fill_between(days_GMES, mean_GMES - 100*variance_GMES, mean_GMES + 100*variance_GMES, color='purple', alpha=0.3)

print(mean_p[0], mean_c[0], mean_DoSS[0], mean_GMES[0])
# print(mean_p[30], mean_GreedyBO[30], mean_c[30], mean_DoSS[30], mean_GMES[30])
# print(mean_p[60], mean_GreedyBO[60], mean_c[60], mean_DoSS[60], mean_GMES[60])

plt.xlim(-5, 100)

plt.xlabel('Iterations')
plt.ylabel('WRMSE')
plt.legend()
plt.grid(True)

plt.savefig("/home/clp/catkin_ws/src/source_seeking/record/img/trend.png")
plt.show()




