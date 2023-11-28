import matplotlib.pyplot as plt

# Step 1: Read Data from File
data = []
with open('temp.txt', 'r') as file:
    for line in file:
        # Convert each line to a float and append to the data list
        data.append(float(line.strip()))

# Step 2: Plot the Data
plt.plot(data)
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Line Plot of Data from temp.txt')
plt.show()
