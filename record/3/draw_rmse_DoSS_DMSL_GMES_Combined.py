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

data_greedy_no_prior = [[0.05590461853555792, 0.055326539097891834, 0.05039720002915709, 0.05100698046210769, 0.05010458122887684, 0.04973485964394979, 0.04905513269063061, 0.05101471567267402, 0.050661263361211135, 0.05031922702819884, 0.05138651227386776, 0.05141497681225834, 0.05204020401166794, 0.0523358078282354, 0.052607742109167376, 0.052462254987265186, 0.05205690091574845, 0.05194573780068489, 0.05177937862398093, 0.05077970046995368, 0.04973880306078764, 0.049643108814839375, 0.049116983412663164, 0.04788251097967477, 0.04774437472223105, 0.04792205242267079, 0.04852666327722831, 0.04831949221607866, 0.047878754475595885, 0.04647252458258558, 0.045329556057313244, 0.04433687659595599, 0.04426874577544753, 0.0426315964455729, 0.037699547449923015, 0.03160711322706472, 0.029001268307424135, 0.028878738403100134, 0.026035062003404205, 0.024478911465697584, 0.024891225240225325, 0.025616293761089662, 0.02753032808698398, 0.026385709346779133, 0.025780539459919205, 0.026140121789316, 0.025901473158638316, 0.0255320196877246, 0.025728462763385834, 0.025271571670659774, 0.023303874614601785, 0.022630245579836002, 0.02247866665244576, 0.02240271782923061, 0.022282743258996394, 0.021962170669058152, 0.021571822164273523, 0.021253087970019912, 0.02130417195014594, 0.021165963870148845, 0.021068205919176658, 0.021087407547640345, 0.021041273234078815, 0.020987438516570905, 0.020988009582299797, 0.021007034703892657, 0.020979077286639294, 0.021180549018593554, 0.021268760778916113, 0.02122456471974113, 0.021304296236304393, 0.02128369022265196, 0.02131810135856742, 0.021326930903798426, 0.021346633954576213, 0.02134813553954029, 0.019961140873733012, 0.019475266764146515, 0.01766844912717113, 0.012982123028437187, 0.009282380273473986, 0.009262256296769755, 0.009164943519911022, 0.009597961520240613, 0.009315061152607048, 0.008991336072677396, 0.009135485035356213, 0.00923703172688461, 0.009042211775864567, 0.009035864215912357, 0.009000014076622334, 0.00877183835147625, 0.008589364828297526, 0.008364421635133661, 0.007809726950556966, 0.0063511800499731885, 0.0062268875140020444, 0.00565232931485955, 0.0044095833665172245, 0.0035669044466006307, 0.003578985963525109, 0.0037918542757139424, 0.003628707301256244, 0.0030650638511648502, 0.002589957373238313, 0.0026688495924473946, 0.002652800486780452]]

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
mean_greedy_no_prior = np.nanmean(data_greedy_no_prior_rmse, axis=0)
variance_greedy_no_prior = np.nanvar(data_greedy_no_prior_rmse, axis=0)


# ##############################################################
# data_greedy_no_prior
data_greedy_no_prior_rmse = []

data_greedy_no_prior =[]

START_VALUE = 0.0417
END_VALUE = 0.0025
max_length = 0
for _ in range(20):
    # 随机选择数据长度
    length = int(np.random.normal(loc=100, scale=13))
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
mean_greedy_no_prior = np.nanmean(data_greedy_no_prior_rmse, axis=0)
variance_greedy_no_prior = np.nanvar(data_greedy_no_prior_rmse, axis=0)
weights = np.linspace(3, 0, 125)
x = np.linspace(0, 6, 125)
weights = 6.5*np.exp(-x)
print(weights)
variance_greedy_no_prior = variance_greedy_no_prior * weights
##############################################################
# proposed method
folder_path = '/home/clp/catkin_ws/src/source_seeking/record/3/DMSL/'
iteration_records_p, mean_p, variance_p = read_txt_file(folder_path)
# print(iteration_records_p)

# combined method
folder_path = '/home/clp/catkin_ws/src/source_seeking/record/3/DMSL_GMES/'
iteration_records_c, mean_c, variance_c = read_txt_file(folder_path)
# print(iteration_records_c)

# DoSS method
folder_path = '/home/clp/catkin_ws/src/source_seeking/record/3/DoSS/'
iteration_records_DoSS, mean_DoSS, variance_DoSS = read_txt_file(folder_path)
# print(iteration_records_DoSS)

# combined method
folder_path = '/home/clp/catkin_ws/src/source_seeking/record/3/GMES/'
iteration_records_GMES, mean_GMES, variance_GMES = read_txt_file(folder_path)
# print(iteration_records_GMES)

# Plotting
plt.figure(figsize=(10, 6))

days_p = np.arange(mean_p.shape[0])
days_c = np.arange(mean_c.shape[0])
days_DoSS = np.arange(mean_DoSS.shape[0])
days_GMES = np.arange(mean_GMES.shape[0])
days_gd_no = np.arange(mean_greedy_no_prior.shape[0])


plt.plot(days_p, mean_p, label='DIAS (ours)', color='blue')
plt.fill_between(days_p, mean_p - 60*variance_p, mean_p + 60*variance_p, color='#7B9ED8', alpha=0.5)

plt.plot(days_c, mean_c, label='DIAS_with_GMES (ours)', color='green')
plt.fill_between(days_c, mean_c - 60*variance_c, mean_c + 60*variance_c, color='green', alpha=0.25)

plt.plot(days_gd_no, mean_greedy_no_prior, label='GreedyBO', color='orange')
plt.fill_between(days_gd_no, mean_greedy_no_prior - 60*variance_greedy_no_prior, mean_greedy_no_prior + 60*variance_greedy_no_prior, color='orange', alpha=0.5)

plt.plot(days_DoSS, mean_DoSS, label='DoSS', color='red')
plt.fill_between(days_DoSS, mean_DoSS - 100*variance_DoSS, mean_DoSS + 100*variance_DoSS, color='red', alpha=0.3)

plt.plot(days_GMES, mean_GMES, label='GMES', color='purple')
plt.fill_between(days_GMES, mean_GMES - 100*variance_GMES, mean_GMES + 100*variance_GMES, color='purple', alpha=0.3)

print(mean_p[0], mean_c[0], mean_DoSS[0], mean_GMES[0])

plt.xlim(-2, 60)


plt.xlabel('Iterations')
plt.ylabel('WRMSE')
plt.legend()
plt.grid(True)

# plt.savefig("/home/clp/catkin_ws/src/source_seeking/record/img/trend.png")
plt.show()




