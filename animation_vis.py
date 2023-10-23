import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from src.math_lib.VectorOperations import euler_dcm_body_to_inertial, euler_dcm_inertial_to_body

"""

Be able to animate the position and direction vectors of the aircraft in 3D space


Keep in mind this is all in NED frames
"""

if __name__=="__main__":

    # Load your data from CSV or any other source
    # Assuming your data is loaded into a DataFrame named 'data'
    # data = pd.read_csv('your_data.csv')
    data = pd.read_csv("rk45_states.csv")
    print(data)    
    
    # wing span of the aircraft to the left and right of center line of the aircraft
    wing_span = 5 #meters 
    fuse_length = 10 #meters

    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Function to update the plot for each frame of the animation
    def update(frame):
        ax.cla()  # Clear the previous frame
        
        # Plot positions
        ax.scatter(data['x'][frame], data['y'][frame], data['z'][frame], color='b', label='Position')
        
        # Plot body frame direction vector (assuming unit vectors for simplicity)
        body_frame_vector = np.array([data['u'][frame], data['v'][frame], data['w'][frame]])

        # Compute the inertial frame direction vector
        dcm_body_to_inertial = euler_dcm_body_to_inertial(data['phi'][frame], 
                                                          data['theta'][frame], 
                                                          data['psi'][frame])

        right_wing = np.array([0, wing_span/2 , 0])
        left_wing = np.array([0, -wing_span/2 ,0])
        
        aircraft_position = np.array([data['x'][frame], data['y'][frame], data['z'][frame]])
        right_wing_inertial = aircraft_position + np.dot(dcm_body_to_inertial, right_wing)
        left_wing_inertial = aircraft_position + np.dot(dcm_body_to_inertial, left_wing)

        #draw a quiver from the aircraft position to the right wing
        ax.quiver(data['x'][frame], data['y'][frame], data['z'][frame],
                right_wing_inertial[0] - data['x'][frame], 
                right_wing_inertial[1] - data['y'][frame], 
                right_wing_inertial[2] - data['z'][frame], 
                color='green', label='Right Wing', linestyle='--')
        
        #draw a quiver from the aircraft position to the left wing
        ax.quiver(data['x'][frame], data['y'][frame], data['z'][frame],
                left_wing_inertial[0] - data['x'][frame], 
                left_wing_inertial[1] - data['y'][frame], 
                left_wing_inertial[2] - data['z'][frame], 
                color='green', linestyle='--')
        
        #do the front and back of the aircraft
        front = np.array([fuse_length/2, 0, 0])
        back = np.array([-fuse_length/2, 0, 0])

        front_inertial = aircraft_position + np.dot(dcm_body_to_inertial, front)
        back_inertial = aircraft_position + np.dot(dcm_body_to_inertial, back)

        #draw a quiver from the aircraft position to the front
        ax.quiver(data['x'][frame], data['y'][frame], data['z'][frame],
                front_inertial[0] - data['x'][frame], 
                front_inertial[1] - data['y'][frame], 
                front_inertial[2] - data['z'][frame], 
                color='red', label='Front', linestyle='--')
        
        #draw a quiver from the aircraft position to the back
        #plot without the quiver
        ax.plot([data['x'][frame], back_inertial[0]], 
                [data['y'][frame], back_inertial[1]], 
                [data['z'][frame], back_inertial[2]], 
                color='red', linestyle='--', linewidth=2)
        

        
        #ax.scatter(right_wing_inertial[0], right_wing_inertial[1], right_wing_inertial[2], color='red', label='Right Wing')

        inertial_vel = dcm_body_to_inertial @ body_frame_vector
        #normalize inertial vel
        inertial_vel = inertial_vel / np.linalg.norm(inertial_vel)
        inertial_vel = inertial_vel * 10 
        
        ax.quiver(data['x'][frame], data['y'][frame], data['z'][frame], 
                inertial_vel[0], inertial_vel[1], inertial_vel[2], 
                color='blueviolet', label='Direction Vector')

        

        # Set plot limits if needed
        ax.set_xlim([min(data['x']), max(data['x'])])
        ax.set_ylim([min(data['y']), max(data['y'])])
        ax.set_zlim([min(data['z']), max(data['z'])])
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set title
        ax.set_title(f'Frame: {frame}')
        
        # Set legend
        ax.legend()

    # Animate the frames
    num_frames = len(data)
    ani = FuncAnimation(fig, update, frames=num_frames, interval=1, repeat=False,
                        blit=False)

    # If you want to save the animation to a file, uncomment the next line
    # ani.save('animation.gif', writer='imagemagick', fps=10)

    # Show the animation
    plt.show()