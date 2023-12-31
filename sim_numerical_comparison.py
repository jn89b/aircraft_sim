import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from src.aircraft.AircraftDynamics import AircraftDynamics, AircraftDynamicsV2
from src.aircraft.Aircraft import AircraftInfo
from src.Utils import get_airplane_params

"""

Just evaluating the numerical approximations between the using 
Euler's method and Runge-Kutta 4th order method.

"""

plt.close('all')
if __name__=="__main__":
    #check if csv file exists

    df = pd.read_csv("SIM_Plane_h_vals.csv")
    airplane_params = get_airplane_params(df)

    use_csv_for_init = False
    
    if use_csv_for_init == False:
            
        init_states = {'x':0.0, 'y':0.0, 'z':0.0,
                    'u':25.0, 'v':0.0, 'w':0.0,
                    'phi':0.0, 'theta':0.0, 'psi':0.0,
                    'p':0.0, 'q':0.0, 'r':0.0}
        
        #a negatitive elevator command makes it go up 
        controls = {'delta_e':np.deg2rad(10.0), 
                    'delta_a':np.deg2rad(0.0), 
                    'delta_r':np.deg2rad(0.0), 
                    'delta_t':0.4}
        
    else:
        #load csv file
        states_df = pd.read_csv("final_states.csv")
        #skip the first column
        states_df = states_df.iloc[:,1:]
        init_states = states_df.iloc[-1].to_dict()
        
        controls_df = pd.read_csv("final_controls.csv")
        controls_df = controls_df.iloc[:,1:]
        controls = controls_df.iloc[-1].to_dict()
        
        
        init_states = {'x':0.0, 'y':0.0, 'z':0.0,
                    'u':25.0, 'v':0.0, 'w':0.0,
                    'phi':0.0, 'theta':np.deg2rad(0.0306), 'psi':0.0,
                    'p':0.0, 'q':0.0, 'r':0.0}
        
        controls = {'delta_e':np.deg2rad(0.0), 
                    'delta_a':np.deg2rad(0.0), 
                    'delta_r':np.deg2rad(0.0), 
                    'delta_t':0.128}

    aircraft_info_euler = AircraftInfo(
        airplane_params,
        init_states,
        controls)
    
    aircraft_info_rk = AircraftInfo(
        airplane_params,
        init_states,
        controls)
    
    aircraft_dynamics_eulers = AircraftDynamics(aircraft_info_euler)
    aircraft_dynamics_rk = AircraftDynamics(aircraft_info_rk)

    dt = 0.01        
    t_init = 0.0
    t_final = 5.0
    N = int((t_final - t_init) / dt)
    N = 20
    # N = 100

    input_aileron = controls['delta_a']
    input_elevator = controls['delta_e']
    input_rudder = controls['delta_r']
    input_throttle = controls['delta_t']

    euler_states = []
    rk_states = []

    use_linear_csv = False
    if use_linear_csv == True:
        controls_df = pd.read_csv("MPC_Plane_controls.csv")
        states_df = pd.read_csv("MPC_Plane_states.csv")

    for i in range(N):
        if use_linear_csv == True:
            input_aileron = controls_df['delta_a'][i]
            input_elevator = controls_df['delta_e'][i]
            input_rudder = controls_df['delta_r'][i]
            input_throttle = controls_df['delta_t'][i]
        
        new_states_eulers = aircraft_dynamics_eulers.eulers(
            input_aileron,
            input_elevator,
            input_rudder,
            input_throttle,
            aircraft_info_euler.states,
            dt
        )
        
        aircraft_info_euler.update_states(new_states_eulers)
        euler_states.append(new_states_eulers)
        
        new_states_rk = aircraft_dynamics_rk.rk45(
            input_aileron,
            input_elevator,
            input_rudder,
            input_throttle,
            aircraft_info_rk.states,
            dt
        )
        
        aircraft_info_rk.update_states(new_states_rk)
        rk_states.append(new_states_rk)

    euler_states = pd.DataFrame(euler_states)
    rk_states = pd.DataFrame(rk_states)

    #set column names
    euler_states.columns = ['x','y','z',
                            'u','v','w',
                            'phi','theta',
                            'psi','p','q','r']
    
    euler_states['z'] = -euler_states['z']
    
    rk_states.columns = ['x','y','z',
                         'u','v','w',
                         'phi','theta','psi',
                         'p','q','r']

    #save rk45 states to csv
    rk_states.to_csv("rk45_states.csv", index=False)


    print(euler_states)
    print("RK45", rk_states)

    # # plot the results
    import matplotlib.pyplot as plt
    
    #3d plot 
    from mpl_toolkits.mplot3d import Axes3D
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':'3d'})
    # ax.plot(euler_states['x'], euler_states['y'], -euler_states['z'], 'o-', 
    #         label='Euler',)
    # ax.plot(rk_states['x'], rk_states['y'], -rk_states['z'], 'x-' ,
    #         label='RK45')
    ax.legend()
    #plot attitudes in euler angles in a subplot
    fig1, ax1 = plt.subplots(3,1,sharex=True)
    time_vec = np.linspace(0, t_final, N)
    ax1[0].plot(time_vec, np.rad2deg(euler_states['phi']), label='Euler')
    ax1[0].plot(time_vec, np.rad2deg(rk_states['phi']), label='RK45')

    ax1[1].plot(time_vec, np.rad2deg(euler_states['theta']), label='Euler')
    ax1[1].plot(time_vec, np.rad2deg(rk_states['theta']), label='RK45')

    ax1[2].plot(time_vec, np.rad2deg(euler_states['psi']), label='Euler')
    ax1[2].plot(time_vec, np.rad2deg(rk_states['psi']), label='RK45')
    ax1[2].set_xlabel('Time (s)')
    ax1[0].set_ylabel('Roll (deg)')
    ax1[1].set_ylabel('Pitch (deg)')
    ax1[2].set_ylabel('Yaw (deg)')
    
    #plot 3d position
    fig2, ax2 = plt.subplots(1,1,subplot_kw={'projection':'3d'})
    
    ax2.plot(euler_states['x'], euler_states['y'], euler_states['z'], 'o-',
                label='Euler')  
    if use_linear_csv == True:
        ax2.plot(states_df['x'], states_df['y'], -states_df['h'], 'x-',
                label='Linear')
        
    ax2.legend()
    
    #format axis to be same scale
    max_range = np.array([euler_states['x'].max()-euler_states['x'].min(),
                            euler_states['y'].max()-euler_states['y'].min(),
                            euler_states['z'].max()-euler_states['z'].min()]).max() / 2.0
    
    mid_x = (euler_states['x'].max()+euler_states['x'].min()) * 0.5
    mid_y = (euler_states['y'].max()+euler_states['y'].min()) * 0.5
    mid_z = (euler_states['z'].max()+euler_states['z'].min()) * 0.5
    
    # ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    # ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    # ax2.set_zlim(mid_z - max_range, mid_z + max_range)
    
    #plot body velocities
    fig3, ax3 = plt.subplots(3,1,sharex=True)
    ax3[0].plot(time_vec, euler_states['u'], label='Euler')
    # ax3[0].plot(time_vec, rk_states['u'], label='RK45')
    if use_linear_csv == True:
        ax3[0].plot(time_vec, states_df['u'], label='Linear')
    
    ax3[0].set_ylabel('u (m/s)')
    ax3[0].legend()
    
    ax3[1].plot(time_vec, euler_states['v'], label='Euler')
    ax3[1].plot(time_vec, rk_states['v'], label='RK45')
    ax3[1].set_ylabel('v (m/s)')
    ax3[2].plot(time_vec, euler_states['w'], label='Euler')
    ax3[2].plot(time_vec, rk_states['w'], label='RK45')
    ax3[2].set_ylabel('w (m/s)')
    ax3[2].set_xlabel('Time (s)')
    

    def update(frame):
        ax.cla()  # Clear the previous frame
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Aircraft Animation')

        #keep last 100 frames
        frame_len = 150

        if frame > frame_len:
            pos_bounds = 0.5
            x_positions = euler_states['x'][frame-frame_len:frame]
            y_positions = euler_states['y'][frame-frame_len:frame]
            z_positions = euler_states['z'][frame-frame_len:frame]

            # Set axis limits based on the current frame's positions
            ax.set_xlim(min(x_positions) - pos_bounds, max(x_positions) + pos_bounds)
            ax.set_ylim(min(y_positions) - pos_bounds, max(y_positions) + pos_bounds)
            ax.set_zlim(min(z_positions) - pos_bounds, max(z_positions) + pos_bounds)
        else:
            # plot everything
            x_positions = euler_states['x'][0:frame]
            y_positions = euler_states['y'][0:frame]
            z_positions = euler_states['z'][0:frame]

        #modulus of the frame
        frame = frame % frame_len

        alpha_vector = np.linspace(0, 1, len(x_positions))
        # print(alpha_vector)
        #multiply the color blue to the alpha vector
        # color_vector = []
        # for alpha in alpha_vector:
        #     color_vector.append((0,0,1,alpha))

        # Plot the new positions for each frame
        # for i in range(len(x_positions)):
            # ax.scatter(x_positions[i], y_positions[i], z_positions[i], color='b', alpha=alpha_vector[i])
        ax.plot(x_positions, y_positions, z_positions, 'o-', color='b', label='Euler and RK45')
        ax.legend()

    # Function to draw the aircraft orientation
    def draw_aircraft(ax, x, y, z, heading, pitch, roll):
        # Define aircraft body dimensions (for example)
        body_length = 5
        body_width = 1
        body_height = 1

        # Calculate body orientation
        rotation_matrix = np.array([
            [np.cos(heading) * np.cos(pitch),
            np.cos(heading) * np.sin(pitch) * np.sin(roll) - np.sin(heading) * np.cos(roll),
            np.cos(heading) * np.sin(pitch) * np.cos(roll) + np.sin(heading) * np.sin(roll)],
            [np.sin(heading) * np.cos(pitch),
            np.sin(heading) * np.sin(pitch) * np.sin(roll) + np.cos(heading) * np.cos(roll),
            np.sin(heading) * np.sin(pitch) * np.cos(roll) - np.cos(heading) * np.sin(roll)],
            [-np.sin(pitch),
            np.cos(pitch) * np.sin(roll),
            np.cos(pitch) * np.cos(roll)]
        ])

        # Define body vertices in body-fixed frame
        body_vertices = np.array([
            [body_length / 2, 0, 0],
            [-body_length / 2, body_width / 2, 0],
            [-body_length / 2, -body_width / 2, 0],
            [-body_length / 2, -body_width / 2, -body_height],
            [-body_length / 2, body_width / 2, -body_height],
            [body_length / 2, 0, -body_height]
        ])

        # Rotate body vertices to world frame
        rotated_vertices = np.dot(body_vertices, rotation_matrix.T)
        rotated_vertices += np.array([x, y, z])  # Translate to the aircraft position

        # Draw the aircraft body
        ax.plot3D(rotated_vertices[:, 0], rotated_vertices[:, 1], rotated_vertices[:, 2], color='b')

    # Create the animation
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':'3d'})

    ani = FuncAnimation(fig, update, frames=len(euler_states), interval=30)
    plt.show()
