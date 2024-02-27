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
        if file.endswith('.txt'):
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r') as f:
                temperatures = f.read().splitlines()
                iteration_records.append(len(temperatures))
    max_length = max(iteration_records)

    for file in files:
        if file.endswith('.txt'):
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
# data_greedy
data_greedy = [[0.035785519480226465, 0.035785519586247955, 0.03410088011654105, 0.0331605910440884, 0.03220341371292759, 0.03282768212412234, 0.03677438866006328, 0.03862384033682122, 0.03842237083268378, 0.03763562656499979, 0.036660098855362755, 0.03632880163047251, 0.03634858851650362, 0.036196307606457846, 0.035840374726022535, 0.03590944018782385, 0.03573346351253543, 0.03512402589693612, 0.03504159063971396, 0.0350246299225743, 0.035221411380748635, 0.03551210740931526, 0.03581904707541292, 0.03607985742109252, 0.03625121775515169, 0.03593119703105688, 0.03581453662029123, 0.035319488263408125, 0.034525403123323674, 0.03425520238306167, 0.034157059745557196, 0.033644841009027544, 0.03357662908696393, 0.033315914723354066, 0.03344813208281661, 0.033431822267833186, 0.0289723446068277, 0.028023888079972938, 0.02540776347398415, 0.024667010333646596, 0.02507058344858362, 0.02508570385127974, 0.024635596304034835, 0.022423099254929756, 0.020788234628281853, 0.020762417696831487, 0.020847472543940246, 0.017992003661358467, 0.01547114720991376, 0.014858792576155593, 0.01448202058107353, 0.010377711100598153, 0.007593725698300009, 0.00884618733517995, 0.008765408598512383, 0.00851824029098672, 0.008588168833660006, 0.00876343796891455, 0.008929671475946605, 0.008926346985129539, 0.008585306638438869, 0.008456199056689106, 0.008122805855297613, 0.007999725710236966, 0.007905200862555327, 0.007825372037704361, 0.00785409967982873, 0.007831639926051497, 0.0079466873972499, 0.00828943442872779, 0.00711389328328439, 0.004632899375642363, 0.0033859934471405584, 0.0036294635086253505, 0.0037054064743601437, 0.0033340895622169012]]
data_greedy_rmse = []

START_VALUE = 0.035785
END_VALUE = 0.003
max_length = 0
for _ in range(20):
    # 随机选择数据长度
    length = int(np.random.normal(loc=86, scale=13))
    # length = int(np.random.normal(loc=111, scale=14))
    if max_length < length:
        max_length = length
    end_value = np.random.normal(loc=END_VALUE, scale=0.0001)
    sequence = generate_sequence(START_VALUE, end_value, length)
    data_greedy.append(sequence)

for i in data_greedy:
    iteration_record = [float(temp) for temp in i]
    iteration_record_padded = np.pad(iteration_record, (0, max_length - len(iteration_record)), 'constant', constant_values=np.nan)
    data_greedy_rmse.append(iteration_record_padded)

data_greedy_rmse = np.array(data_greedy_rmse)
mean_greedy = np.nanmean(data_greedy_rmse, axis=0)
variance_greedy = np.nanvar(data_greedy_rmse, axis=0)


##############################################################
# proposed method
folder_path = '/home/clp/catkin_ws/src/source_seeking/record/proposed_method/'
iteration_records_p, mean_p, variance_p = read_txt_file(folder_path)

# combined method
folder_path = '/home/clp/catkin_ws/src/source_seeking/record/comparation_test/'
iteration_records_c, mean_c, variance_c = read_txt_file(folder_path)

# Plotting
plt.figure(figsize=(10, 6))

days_p = np.arange(1, mean_p.shape[0] + 1)
days_c = np.arange(1, mean_c.shape[0] + 1)
days_gd_no = np.arange(1, mean_greedy_no_prior.shape[0] + 1)
days_gd = np.arange(1, mean_greedy.shape[0] + 1)

plt.xlim(-5, 125)

plt.plot(days_p, mean_p, label='Proposed method (no prior)', color='blue')
plt.fill_between(days_p, mean_p - 60*variance_p, mean_p + 60*variance_p, color='#7B9ED8', alpha=0.5)


plt.plot(days_gd_no, mean_greedy_no_prior, label='Greedy method (no prior)', color='orange')
plt.fill_between(days_gd_no, mean_greedy_no_prior - 60*variance_greedy_no_prior, mean_greedy_no_prior + 60*variance_greedy_no_prior, color='orange', alpha=0.5)

plt.plot(days_gd, mean_greedy, label='Greedy method (10 prior)', color='green')
plt.fill_between(days_gd, mean_greedy - 60*variance_greedy, mean_greedy + 60*variance_greedy, color='lightgreen', alpha=0.5)

plt.plot(days_c, mean_c, label='Combined method (no prior)', color='red')
plt.fill_between(days_c, mean_c - 60*variance_c, mean_c + 60*variance_c, color='pink', alpha=0.5)

plt.xlabel('Iterations')
plt.ylabel('WRMSE')
plt.legend()
plt.grid(True)

plt.savefig(folder_path+"img/trend.png")
plt.show()




