import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from src.aircraft.AircraftDynamics import AircraftDynamics
from src.aircraft.Aircraft import AircraftInfo
from src.Utils import get_airplane_params


"""
Just evaluating the numerical approximations between the using 
Euler's method and Runge-Kutta 4th order method.
"""

if __name__=="__main__":
    #check if csv file exists

    df = pd.read_csv("SIM_Plane_h_vals.csv")
    airplane_params = get_airplane_params(df)
    
    init_states = {'x':0.0, 'y':0.0, 'z':0.0,
                   'u':0.0, 'v':0.0, 'w':0.0,
                   'phi':0.0, 'theta':0.0, 'psi':0.0,
                   'p':0.0, 'q':0.0, 'r':0.0}
    
    controls = {'delta_e':np.deg2rad(25), 
                'delta_a':np.deg2rad(5), 
                'delta_r':0.0, 
                'delta_t':150.0}
    
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
    t_final = 10
    N = int((t_final - t_init) / dt)
    print(N)

    input_aileron = controls['delta_a']
    input_elevator = controls['delta_e']
    input_rudder = controls['delta_r']
    input_throttle = controls['delta_t']

    euler_states = []
    rk_states = []

    for i in range(N):

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
    euler_states.columns = ['x','y','z','u','v','w','phi','theta','psi','p','q','r']
    rk_states.columns = ['x','y','z','u','v','w','phi','theta','psi','p','q','r']

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
    ax1[0].plot(np.rad2deg(euler_states['phi']), label='Euler')
    ax1[0].plot(np.rad2deg(rk_states['phi']), label='RK45')

    ax1[1].plot(np.rad2deg(euler_states['theta']), label='Euler')
    ax1[1].plot(np.rad2deg(rk_states['theta']), label='RK45')

    ax1[2].plot(np.rad2deg(euler_states['psi']), label='Euler')
    ax1[2].plot(np.rad2deg(rk_states['psi']), label='RK45')
    ax1[2].set_xlabel('Time (s)')
    ax1[0].set_ylabel('Roll (deg)')
    ax1[1].set_ylabel('Pitch (deg)')
    ax1[2].set_ylabel('Yaw (deg)')

    def update(frame):
        ax.cla()  # Clear the previous frame
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Aircraft Animation')

        #keep last 100 frames
        frame_len = 50

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
        ax.plot(x_positions, y_positions, z_positions, 'o-', color='b', label='Euler and RK45')
        ax.legend()
        
        return ax

    # Create the animation
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':'3d'})

    # Function to initialize the animation (blit needs this function)
    def init():
        # Remove the initial plot (or any other objects you want to blit)
        # initial_plot.pop(0).remove()
        return ax

    uas_paths = []
    uas_paths.append([x_positions, y_positions, z_positions])
    
    lines = ax.plot([], [], [], linewidth=2)[0] 
                        for _ in range(len(uas_paths))

    ani = FuncAnimation(fig, update, init_func=init, 
                        frames=len(euler_states), interval=30, blit=True)
    plt.show()
