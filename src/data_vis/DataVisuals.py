import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from src.math_lib.VectorOperations import euler_dcm_body_to_inertial, \
    euler_dcm_inertial_to_body


class DataVisualization():
    def __init__(self, data:pd.DataFrame) -> None:
        self.data = data
        
        #this is stupid but I need to do this to get the wing span
        self.wing_span = 5 #meters
        self.fuselage_length = 3 #meters
        self.compute_fuselage_wing_positions()

    def compute_fuselage_wing_positions(self):
        # computes the right and left wing positions in inertial frame
        x_positions = self.data['x'].values
        y_positions = self.data['y'].values
        z_positions = self.data['z'].values

        phi_angles = self.data['phi'].values
        theta_angles = self.data['theta'].values
        psi_angles = self.data['psi'].values

        right_wing_positions = []
        left_wing_positions = []
        front_fuselage_positions = []
        back_fuselage_positions = []

        for i in range(len(x_positions)):
            aircraft_position = np.array([x_positions[i], y_positions[i], z_positions[i]])
            dcm_body_to_inertial = euler_dcm_body_to_inertial(phi_angles[i], 
                                                              theta_angles[i], 
                                                              psi_angles[i])
            right_wing = np.array([0, self.wing_span/2 , 0])
            left_wing = np.array([0, -self.wing_span/2 ,0])
            
            right_wing_inertial = aircraft_position + np.dot(dcm_body_to_inertial, right_wing)
            left_wing_inertial = aircraft_position + np.dot(dcm_body_to_inertial, left_wing)

            front_fuselage = np.array([self.fuselage_length/2, 0, 0])
            back_fuselage = np.array([-self.fuselage_length/2, 0, 0])

            right_wing_positions.append(right_wing_inertial)
            left_wing_positions.append(left_wing_inertial)

            front_fuselage_positions.append(aircraft_position + np.dot(dcm_body_to_inertial, front_fuselage))
            back_fuselage_positions.append(aircraft_position + np.dot(dcm_body_to_inertial, back_fuselage))

        #update the dataframes with the right and left wing positions
        self.data['right_wing_x'] = [x[0] for x in right_wing_positions]
        self.data['right_wing_y'] = [x[1] for x in right_wing_positions]
        self.data['right_wing_z'] = [x[2] for x in right_wing_positions]

        self.data['left_wing_x'] = [x[0] for x in left_wing_positions]
        self.data['left_wing_y'] = [x[1] for x in left_wing_positions]
        self.data['left_wing_z'] = [x[2] for x in left_wing_positions]

        self.data['front_fuselage_x'] = [x[0] for x in front_fuselage_positions]
        self.data['front_fuselage_y'] = [x[1] for x in front_fuselage_positions]
        self.data['front_fuselage_z'] = [x[2] for x in front_fuselage_positions]

        self.data['back_fuselage_x'] = [x[0] for x in back_fuselage_positions]
        self.data['back_fuselage_y'] = [x[1] for x in back_fuselage_positions]
        self.data['back_fuselage_z'] = [x[2] for x in back_fuselage_positions]

    def animate(self, interval:int=1):
        # Create the animation
        fig, self.ax = plt.subplots(1,1,subplot_kw={'projection':'3d'})

        #set color to lightblue
        self.ax.set_facecolor((0.529, 0.808, 0.922))

        #set background color to black
        fig.patch.set_facecolor((0.529, 0.808, 0.922))


        ani = FuncAnimation(fig, self.update, frames=len(self.data), interval=interval)

        plt.show()

    def update(self, frame):
        
        self.ax.cla()
        self.ax.scatter(self.data['x'][frame], 
                        self.data['y'][frame], 
                        self.data['z'][frame], color='b', label='Position')
        
        #draw a quiver from the aircraft position to the right wing
        self.ax.quiver(self.data['x'][frame], self.data['y'][frame], self.data['z'][frame],
                self.data['x'][frame] - self.data['right_wing_x'][frame],
                self.data['y'][frame] - self.data['right_wing_y'][frame],
                self.data['z'][frame] - self.data['right_wing_z'][frame],
                color='green', label='Right Wing')
        
        #draw a quiver from the aircraft position to the left wing
        self.ax.quiver(self.data['x'][frame], self.data['y'][frame], self.data['z'][frame],
                self.data['x'][frame] - self.data['left_wing_x'][frame],
                self.data['y'][frame] - self.data['left_wing_y'][frame],
                self.data['z'][frame] - self.data['left_wing_z'][frame],
                color='green', linestyle='--')
        
        #do the front and back of the aircraft
        #draw a quiver from the aircraft position to the front
        self.ax.quiver(self.data['x'][frame], self.data['y'][frame], self.data['z'][frame],
                self.data['front_fuselage_x'][frame] - self.data['x'][frame], 
                self.data['front_fuselage_y'][frame] - self.data['y'][frame], 
                self.data['front_fuselage_z'][frame] - self.data['z'][frame], 
                color='red', label='Front')
        
        #draw a quiver from the aircraft position to the back
        #plot without the quiver
        self.ax.plot([self.data['x'][frame], self.data['back_fuselage_x'][frame]], 
                [self.data['y'][frame], self.data['back_fuselage_y'][frame]], 
                [self.data['z'][frame], self.data['back_fuselage_z'][frame]], 
                color='red', linestyle='-.')
        

        # Plot body frame direction vector (assuming unit vectors for simplicity)
        body_frame_vector = np.array([self.data['u'][frame], self.data['v'][frame], self.data['w'][frame]])

        # Compute the inertial frame direction vector
        dcm_body_to_inertial = euler_dcm_body_to_inertial(self.data['phi'][frame], 
                                                          self.data['theta'][frame], 
                                                          self.data['psi'][frame])


        # set plot limits based on the current frame's positions
        if self.fuselage_length >= self.wing_span:
            max_length = self.fuselage_length
        else:
            max_length = self.wing_span


        inertial_vel = dcm_body_to_inertial @ body_frame_vector
        #normalize inertial vel
        inertial_vel = inertial_vel / np.linalg.norm(inertial_vel)
        inertial_vel = inertial_vel * max_length 
        

        self.ax.quiver(self.data['x'][frame], self.data['y'][frame], self.data['z'][frame], 
                inertial_vel[0], inertial_vel[1], inertial_vel[2], 
                color='blueviolet', label='Direction Vector')


        current_x = self.data['x'][frame]
        current_y = self.data['y'][frame]
        current_z = self.data['z'][frame]

        self.ax.grid(color='white', linestyle='-', linewidth=0.5)

        self.ax.set_xlim(current_x - max_length, current_x + max_length)
        self.ax.set_ylim(current_y - max_length, current_y + max_length)
        self.ax.set_zlim(current_z - max_length, current_z + max_length)

        # self.draw_aircraft_body(frame)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        self.ax.set_title(f'Frame: {frame}')
        
        #set the legend outside the plot
        self.ax.legend(bbox_to_anchor=(1.0,1), loc="upper right")


    








        



    
