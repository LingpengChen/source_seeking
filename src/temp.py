import numpy as np
import matplotlib.pyplot as plt
x_linspace = np.linspace(-3.2,3.2,100)
x_training = np.array([-1.5])
# x_training = np.array([-3, -1.5, 0, 1.5, 2.5, 2.7])

y_training = np.sin(x_training)
# plt.plot(x_linspace, np.sin(x_linspace))
# plt.scatter(x_training, y_training)
# plt.show()

def squared_exponential_kernel(x1, x2):
    if x1.ndim == 1 and x2.ndim == 1:
        x1 = x1.reshape(-1, 1)
        x2 = x2.reshape(-1, 1)
    from scipy.spatial.distance import cdist
    dx = cdist(x1, x2)
    return np.exp(-(dx ** 2) / 2)

x_testing = np.linspace(-3.2, 3.2, 30)
y_testing = np.sin(x_testing)
K = squared_exponential_kernel(x_training, x_training)
K_star = squared_exponential_kernel(x_testing, x_training)
y_predict = K_star.dot(np.linalg.inv(K)).dot(y_training)
# plt.plot(x_linspace, np.sin(x_linspace))
plt.scatter(x_training, y_training)
plt.scatter(x_testing, y_predict)
plt.show()
MSE = np.mean((y_predict-y_testing)**2)
print(f'MSE = {MSE}')