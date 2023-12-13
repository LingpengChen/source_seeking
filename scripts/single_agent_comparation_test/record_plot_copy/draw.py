import os
import matplotlib.pyplot as plt

def read_data_from_file(file_path):
    """读取文件中的数据并转换为列表"""
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Convert each line to a float and append to the data list
            data.append(float(line.strip()))
    return data
    
def plot_data(folder_path):
    """读取文件夹中所有txt文件的数据，并绘制折线图"""
    file_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])
    for file_name in file_names:
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            data = read_data_from_file(file_path)
            plt.plot(data, label=file_name)

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Line Graphs from Text Files')
    plt.legend()
    plt.show()

# 用你的文件夹路径替换这里的 'your_folder_path'
folder_path = './'
plot_data(folder_path)
