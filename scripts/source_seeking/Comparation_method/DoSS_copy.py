from matplotlib import pyplot as plt
from environment.environment_and_measurement import Environment
import numpy as np

# Define Field
FIELD_SIZE_X = 10
FIELD_SIZE_Y = 10
x_min = (0, 0)
x_max = (0+FIELD_SIZE_X, 0+FIELD_SIZE_Y)
test_resolution = [10, 10]
X_test_x = np.linspace(x_min[0], x_max[0], test_resolution[0])
X_test_y = np.linspace(x_min[1], x_max[1], test_resolution[1])
X_test_xx, X_test_yy = np.meshgrid(X_test_x, X_test_y)
X = np.vstack(np.dstack((X_test_xx, X_test_yy)))
env = Environment(1)
Y = env.f(X)

def kalman_consensus_filter(z_k, H_k, n_k, V, phi_k, Sigma_k):
        Sigma_k_inv = np.linalg.inv(Sigma_k)
        Y_k = H_k.T @ np.linalg.inv(V) @ H_k
        Sigma_k1 = np.linalg.inv(Sigma_k_inv + Y_k)
        
        y_k = H_k.T @ np.linalg.inv(V) @ z_k
        K_k = Sigma_k1 @ y_k
        
        phi_k1 = phi_k + K_k @ (z_k - H_k @ phi_k)
        
        return phi_k1, Sigma_k1

def Kalman_filter_test():
    I = 3  # Number of robots
    fieldsize = 100  # Size of the field

    # Assuming each robot measures a different point, we create a measurement matrix H_k
    # where each robot has a unique position in the field it measures (represented by 1's in different rows).
    H_k = np.zeros((I, fieldsize))
    np.fill_diagonal(H_k[:, :I], 1)  # Example: each robot measures a different field point, for simplicity

    # Measurement vector z_k for all robots (assuming a unit measurement for simplicity)
    z_k = np.ones((I, 1))

    # Noise vector n_k for all robots (with zero mean and 0.1 standard deviation)
    n_k = np.random.normal(0, 0.1, (I, 1))

    # Noise covariance matrix V (assuming same noise level for all robots)
    V = np.diag([0.01] * I)

    # Initial estimate of the state for the entire field
    phi_k = np.zeros((fieldsize, 1))

    # Initial covariance matrix (assuming initial uncertainty for the entire field)
    Sigma_k = np.eye(fieldsize)

    # Kalman Consensus Filter Function
    

    # Run the Kalman Consensus Filter
    phi_k1, Sigma_k1 = kalman_consensus_filter(z_k, H_k, n_k, V, phi_k, Sigma_k)

    # For brevity, printing only the first 5 elements of phi_k1 and the top-left 5x5 of Sigma_k1
    print("Updated State Estimate (phi_k1):", phi_k1[:5], sep="\n")
    print("Updated Covariance Matrix (Sigma_k1):", Sigma_k1[:5, :5], sep="\n")

if __name__ == '__main__':
    # source_value = sampling(SOURCE)
    Kalman_filter_test()
    # # 设置图像的长宽比为一致
    plt.gca().set_aspect('equal', adjustable='box')

    plt.contourf(X_test_xx, X_test_yy, Y.reshape(test_resolution), cmap='coolwarm',  edgecolor='none', levels=100)
    plt.xticks([])
    plt.yticks([])
    # 
    plt.show()
    
    
    