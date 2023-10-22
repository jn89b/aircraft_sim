import pandas as pd
import numpy as np

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
                'delta_a':np.deg2rad(0), 
                'delta_r':0.0, 
                'delta_t':15.0}
    
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

    print(aircraft_info_euler.states)
  
    dt = 0.01        
    t_init = 0.0
    t_final = 20
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

        print(type(new_states_rk[7]) )

        rk_states.append(new_states_rk)

    euler_states = pd.DataFrame(euler_states)
    rk_states = pd.DataFrame(rk_states)

    #set column names
    euler_states.columns = ['x','y','z','u','v','w','phi','theta','psi','p','q','r']
    rk_states.columns = ['x','y','z','u','v','w','phi','theta','psi','p','q','r']

    print(euler_states)

    # # plot the results
    import matplotlib.pyplot as plt
    
    #3d plot 
    from mpl_toolkits.mplot3d import Axes3D
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':'3d'})
    ax.plot(euler_states['x'], euler_states['y'], -euler_states['z'], 'o-', 
            label='Euler',)
    ax.plot(rk_states['x'], rk_states['y'], -rk_states['z'], 'x-' ,
            label='RK45')
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

    plt.show() 

    
