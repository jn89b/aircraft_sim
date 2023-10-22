from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import numpy as np

# Initialize DCM as an identity matrix
dcm = np.identity(3)

# Example angular velocity vector (rate values)

p = np.deg2rad(0) # rates in rad/s
q = np.deg2rad(0)
r = np.deg2rad(90)

gyro = np.array([p, q, r])  

# Time interval and number of frames for the animation
delta_time = 0.2
num_frames = 100

# Function to update the DCM based on gyro readings
def update(frame):
    global dcm
    # Rotate the DCM based on gyro readings
    # rotation = R.from_rotvec(gyro * delta_time * (frame + 1))
    rotation = R.from_rotvec(gyro * delta_time)
    dcm = rotation.as_matrix().dot(dcm)
    # Plot the rotated coordinate system
    ax.cla()
    ax.quiver(0, 0, 0, dcm[0, 0], dcm[1, 0], dcm[2, 0], color='r', label='X-axis')
    ax.quiver(0, 0, 0, dcm[0, 1], dcm[1, 1], dcm[2, 1], color='g', label='Y-axis')
    ax.quiver(0, 0, 0, dcm[0, 2], dcm[1, 2], dcm[2, 2], color='b', label='Z-axis')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('DCM Rotation Animation')
    ax.legend()

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the animation
animation = FuncAnimation(fig, update, frames=num_frames, interval=delta_time * 1000, blit=False)



# Display the animation
plt.show()
