import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load STL file
stl_mesh = mesh.Mesh.from_file('yf23.stl')
print(stl_mesh)

# Define animation parameters
num_frames = 100  # Number of frames in the animation
rotation_angle = 360  # Rotation angle for one complete revolution (in degrees)

# Create a figure and 3D axis
fig = plt.figure()
ax = mplot3d.Axes3D(fig)

# Function to update the plot for each frame of the animation
def update(frame):
    ax.cla()  # Clear the previous frame

    # Calculate rotation angle for this frame
    rotation_step = rotation_angle / num_frames
    angle = frame * rotation_step
    
    # Apply rotation transformation to the STL mesh
    rotated_mesh = stl_mesh.rotate([0, 0, 1], np.radians(angle))

    # Plot the rotated mesh
    ax.add_collection3d(mplot3d.art3d.Poly3DCollection(rotated_mesh.vectors, color='b'))
    ax.set_xlim([min(rotated_mesh.x), max(rotated_mesh.x)])
    ax.set_ylim([min(rotated_mesh.y), max(rotated_mesh.y)])
    ax.set_zlim([min(rotated_mesh.z), max(rotated_mesh.z)])

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Frame: {frame}')

# Animate the frames
ani = FuncAnimation(fig, update, frames=num_frames, interval=50, repeat=False)

# If you want to save the animation to a file, uncomment the next line
# ani.save('animation.gif', writer='imagemagick', fps=20)

# Show the animation
plt.show()