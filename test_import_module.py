import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Create a figure with 2x2 subplots
fig = plt.figure(figsize=(12, 12))

# First 3D subplot
ax1 = fig.add_subplot(221, projection='3d')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('3D Animation 1')

# Second 3D subplot
ax2 = fig.add_subplot(222, projection='3d')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('3D Animation 2')

# First 2D subplot
ax3 = fig.add_subplot(223)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_title('2D Animation 1')

# Second 2D subplot
ax4 = fig.add_subplot(224)
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_title('2D Animation 2')

# Number of frames in the animation
num_frames = 100

# Function to generate random 3D data for animation
def generate_3d_data():
    x = np.random.rand(num_frames)
    y = np.random.rand(num_frames)
    z = np.random.rand(num_frames)
    return x, y, z

# Function to generate random 2D data for animation
def generate_2d_data():
    x = np.random.rand(num_frames)
    y = np.random.rand(num_frames)
    return x, y

# Function to update the first 3D subplot
def update_3d_subplot(frame):
    ax1.cla()  # Clear the previous frame
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    x, y, z = generate_3d_data()
    ax1.scatter(x[:frame], y[:frame], z[:frame], color='b')

# Function to update the second 3D subplot
def update_3d_subplot_2(frame):
    ax2.cla()  # Clear the previous frame
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    x, y, z = generate_3d_data()
    ax2.plot(x[:frame], y[:frame], z[:frame], color='g')

# Function to update the first 2D subplot
def update_2d_subplot(frame):
    ax3.cla()  # Clear the previous frame
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')

    x, y = generate_2d_data()
    ax3.scatter(x[:frame], y[:frame], color='r')

# Function to update the second 2D subplot
def update_2d_subplot_2(frame):
    ax4.cla()  # Clear the previous frame
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')

    x, y = generate_2d_data()
    ax4.plot(x[:frame], y[:frame], color='m')

# Create animations for each subplot
ani1 = FuncAnimation(fig, update_3d_subplot, frames=num_frames, interval=100)
ani2 = FuncAnimation(fig, update_3d_subplot_2, frames=num_frames, interval=100)
ani3 = FuncAnimation(fig, update_2d_subplot, frames=num_frames, interval=100)
ani4 = FuncAnimation(fig, update_2d_subplot_2, frames=num_frames, interval=100)

plt.tight_layout()
plt.show()
