import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from src.math_lib.VectorOperations import euler_dcm_body_to_inertial, \
    euler_dcm_inertial_to_body


class DataVisualization():
    def __init__(self, data:pd.DataFrame, keep_every:int=1) -> None:
        # self.data = data.iloc[::2, :]
        self.data = data
        self.data = self.data.iloc[::keep_every, :]        

        #reorder the index
        self.data = self.data.reset_index(drop=True)

        #keep every 2n
        #self.data = self.data.iloc[::2, :]
        
        print("data shape: ", self.data.shape)

        #this is stupid but I need to do this to get the wing span
        self.wing_span = 5 #meters
        self.fuselage_length = 3 #meters
        self.position_list = self.compute_fuselage_wing_positions()


    def compute_fuselage_wing_positions(self) -> list:
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

        right_wing_positions = np.array(right_wing_positions)
        left_wing_positions = np.array(left_wing_positions)
        front_fuselage_positions = np.array(front_fuselage_positions)
        back_fuselage_positions = np.array(back_fuselage_positions)
        aircraft_position = np.array([x_positions, y_positions, z_positions]).T

        position_list = [aircraft_position, right_wing_positions, left_wing_positions,
                        front_fuselage_positions, back_fuselage_positions]
        
        return position_list


    def animate_local(self, interval:int=1, save:bool=False) -> FuncAnimation:
        # Create the animation
        fig, self.ax = plt.subplots(1,1,subplot_kw={'projection':'3d'})

        #set color to lightblue
        self.ax.set_facecolor((0.529, 0.808, 0.922))

        #set background color to black
        fig.patch.set_facecolor((0.529, 0.808, 0.922))

        ani = FuncAnimation(fig, self.update_local, frames=len(self.data),
                             interval=interval)

        #if save is true, save the animation as a gif
        if save == True:
            print("saving gif")
            ani.save('local_animation.gif', writer='imagemagick', fps=30)
            print("done saving gif")
        # plt.show()
        return ani 

    def update_local(self, frame):
        
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
        body_frame_vector = np.array([self.data['u'][frame], 
                                      self.data['v'][frame], 
                                      self.data['w'][frame]])

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

        span = 15
        self.ax.set_xlim(current_x - (max_length + span), current_x + (max_length + span))
        self.ax.set_ylim(current_y - (max_length + span), current_y + (max_length + span))
        self.ax.set_zlim(current_z - (max_length + 0), current_z + (max_length + 0))

        # self.draw_aircraft_body(frame)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        self.ax.set_title(f'Frame: {frame}')
        
        #set the legend outside the plot
        self.ax.legend(bbox_to_anchor=(1.0,1), loc="upper right")


    def animate_global(self, save:bool=False) -> FuncAnimation:
        """
        Animate global position 
        """

        fig1, self.ax1 = plt.subplots(1,1,subplot_kw={'projection':'3d'})

        lines = [self.ax1.plot([], [], [], linewidth=1)[0] 
                        for _ in range(len(self.position_list))]

        pts = [self.ax1.plot([], [], [], 'o')[0] 
                        for _ in range(len(self.position_list))]
        
        x_list = [x[:,0] for x in self.position_list]
        y_list = [x[:,1] for x in self.position_list]
        z_list = [x[:,2] for x in self.position_list]
    
        #set axis limits
        self.ax1.set_xlim3d([min(x_list[0]), max(x_list[0])])
        self.ax1.set_xlabel('X')

        self.ax1.set_ylim3d([min(y_list[0]), max(y_list[0])])
        self.ax1.set_ylabel('Y')

        self.ax1.set_zlim3d([min(z_list[0]), max(z_list[0])])
        self.ax1.set_zlabel('Z')

        color_list = ['b', 'g', 'g', 'r', 'r']
        
        for i in range(len(pts)):
            pts[i].set_color(color_list[i])
            lines[i].set_color(color_list[i])

        
        def init():
            for line in lines:
                line.set_data([], [])
                line.set_3d_properties([])
            for pt in pts:
                pt.set_data([], [])
                pt.set_3d_properties([])
            return lines + pts
        
        def update(i):
            # we'll step two time-steps per frame.  This leads to nice results.
            # i = (2 * i) % x_t.shape[0]

            # for line, pt, xi in zip(lines, pts, x_t):
            frame = i
            time_span = 100 # len(self.data)
            alpha_vec = np.linspace(0, 1, time_span)



            self.ax1.scatter(self.data['x'][frame], 
                            self.data['y'][frame], 
                            self.data['z'][frame], color='b', label='Position')
            
            #draw a quiver from the aircraft position to the right wing
            self.ax1.quiver(self.data['x'][frame], self.data['y'][frame], self.data['z'][frame],
                    self.data['x'][frame] - self.data['right_wing_x'][frame],
                    self.data['y'][frame] - self.data['right_wing_y'][frame],
                    self.data['z'][frame] - self.data['right_wing_z'][frame],
                    color='green', label='Right Wing')
            
            #draw a quiver from the aircraft position to the left wing
            self.ax1.quiver(self.data['x'][frame], self.data['y'][frame], self.data['z'][frame],
                    self.data['x'][frame] - self.data['left_wing_x'][frame],
                    self.data['y'][frame] - self.data['left_wing_y'][frame],
                    self.data['z'][frame] - self.data['left_wing_z'][frame],
                    color='green', linestyle='--')
            
            #do the front and back of the aircraft
            #draw a quiver from the aircraft position to the front
            self.ax1.quiver(self.data['x'][frame], self.data['y'][frame], self.data['z'][frame],
                    self.data['front_fuselage_x'][frame] - self.data['x'][frame], 
                    self.data['front_fuselage_y'][frame] - self.data['y'][frame], 
                    self.data['front_fuselage_z'][frame] - self.data['z'][frame], 
                    color='red', label='Front')
            
            #draw a quiver from the aircraft position to the back
            #plot without the quiver
            self.ax1.plot([self.data['x'][frame], self.data['back_fuselage_x'][frame]], 
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
            

            self.ax1.quiver(self.data['x'][frame], self.data['y'][frame], self.data['z'][frame], 
                    inertial_vel[0], inertial_vel[1], inertial_vel[2], 
                    color='blueviolet', label='Direction Vector')


            for j, (line,pt) in enumerate(zip(lines,pts)):
                
                if i < time_span:
                    interval = 0
                else:
                    interval = i - time_span
                

                #set lines 
                line.set_data(x_list[j][interval:i], y_list[j][interval:i])
                line.set_3d_properties(z_list[j][interval:i])

                # # # #set points

                #update the alpha based on the interval 
                #get the difference between the current interval and the current frame
                # diff = i - interval
                # alpha = alpha_vec[diff]

                # update the alpha for the points based on the interval
                # for k in range(interval, i):
                #     print(interval)
                #     pts[j].set_alpha(alpha_vec[k])

                pt.set_data(x_list[j][interval:i], y_list[j][interval:i])
                pt.set_3d_properties(z_list[j][interval:i])

                #changing views
                # self.ax.view_init(60, 0.3 * i)
                # self.fig.canvas.draw()


            # fig.canvas.draw()
            return lines + pts
        
        num_frames = len(self.data)
        ani = FuncAnimation(fig1, update, frames=num_frames, interval=1, repeat=True,
                            init_func=init, blit=True)
        
        if save:
            ani.save('global_animation.gif', writer='imagemagick', fps=30)
        
        return ani

        


        



