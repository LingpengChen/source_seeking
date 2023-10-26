import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Set parameters
lambda_ = 1.0  # Diffusion coefficient
c = 1.0        # Specific heat
rho = 1.0      # Density
dt = 0.1      # Time step
dx = 0.1       # Spatial step in x direction
dy = 0.1       # Spatial step in y direction

# Initialize space
x = np.arange(0, 10, dx)
y = np.arange(0, 10, dy)
X, Y = np.meshgrid(x, y)
phi = np.zeros_like(X)

# Add two pollution sources
phi[20:30, 20:30] = 1.0  # Pollution source 1
phi[70:80, 70:80] = 1.0  # Pollution source 2

# Set up the figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Update function for animation
def update(num, phi, fig, ax):
    phi_new = phi.copy()
    for i in range(1, phi.shape[0]-1):
        for j in range(1, phi.shape[1]-1):
            diffusion = lambda_ * c / rho * (phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1] - 4*phi[i, j])
            phi_new[i, j] += diffusion * dt
    ax.plot_surface(X, Y, phi_new, cmap='viridis', rstride=1, cstride=1, alpha=0.6, antialiased=True)
    phi[:] = phi_new[:]

# Animation function
ani = animation.FuncAnimation(fig, update, fargs=(phi, fig, ax), frames=100, interval=50, repeat=False)
plt.show()
