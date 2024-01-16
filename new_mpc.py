import numpy as np
import pandas as pd
import casadi as ca

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from src.mpc.FixedWingMPC import FixedWingMPC
from src.aircraft.AircraftDynamics import AircraftCasadi
from src.Utils import get_airplane_params
from src.data_vis.DataVisuals import DataVisualization

from src.aircraft.AircraftDynamics import AircraftDynamics
from src.aircraft.Aircraft import AircraftInfo
import matplotlib.pyplot as plt

if __name__=="__main__":
    plt.close('all')
    df = pd.read_csv("SIM_Plane_h_vals.csv")
    airplane_params = get_airplane_params(df)
    aircraft_ca = AircraftCasadi(aircraft_params=airplane_params)
    aircraft_ca.set_state_space()

    mpc_constraints = {
        'delta_a_max': np.deg2rad(25),
        'delta_a_min': np.deg2rad(-25),
        'delta_e_max': np.deg2rad(25),
        'delta_e_min': np.deg2rad(-25),
        'delta_r_max': np.deg2rad(25),
        'delta_r_min': np.deg2rad(-25),
        'delta_t_max': 0.65, #newtons
        'delta_t_min': 0.35, #newtons
        'z_max': 20, #m
        'z_min': -20, #m
        'u_max': 25.0, #m/s
        'u_min': 15.0, #m/s
        'v_max': 5, #m/s
        'v_min': -5, #m/s
        'w_max': -2, #m/s
        'w_min':  2, #m/s
        'phi_max': np.deg2rad(45), #rad
        'phi_min': np.deg2rad(-45), #rad
        'theta_max': np.deg2rad(25), #rad
        'theta_min': np.deg2rad(-25), #rad
        'p_max': np.deg2rad(90), #rad/s
        'p_min': np.deg2rad(-90), #rad/s
        'q_max': np.deg2rad(90), #rad/s
        'q_min': np.deg2rad(-90), #rad/s
        'r_max': np.deg2rad(90), #rad/s
        'r_min': np.deg2rad(-90), #rad/s
    }

    #create a diagonal matrix for the weights
    Q = ca.diag([1.0,1.0,0.5, #position
                 0.0,0.0,0.0, #velocity body
                 5.0,10.0,5.0, #euler angles
                 0,0,0]) #angular rates body

    R = ca.diag([2.0,
                 2.0,
                 2.0,
                 2.0]) #control inputs

    mpc_params = {
        'model': aircraft_ca,
        'dt_val': 1/100,
        'N': 5,
        'Q': Q,
        'R': R
    }

    fw_mpc = FixedWingMPC(mpc_params, mpc_constraints)
    
    #load planner states
    planner_states = pd.read_csv("planner_states.csv")

    #set initial conditions
    idx_goal = 1

    init_x = planner_states['x'][0]
    init_y = planner_states['y'][0]
    init_z = planner_states['z'][0]
    init_u = 25
    init_v = 0
    init_w = 0
    
    init_psi = np.arctan2(planner_states['y'][1] - planner_states['y'][0], 
                                     planner_states['x'][1] - planner_states['x'][0])
    init_phi = 0.0
    init_theta = 0.0 
    
    print("initial heading" , np.rad2deg(init_psi))    
    init_theta = np.arctan2(planner_states['z'][1] - planner_states['z'][0],
                            np.sqrt((planner_states['x'][1] - planner_states['x'][0])**2 + \
                            planner_states['y'][1] - planner_states['y'][0])**2)
    
    init_p = 0
    init_q = 0
    init_r = 0

    init_al = np.deg2rad(0)
    init_el = np.deg2rad(-1.06)
    init_rud = 0
    init_throttle = 0.64

    #load up planner states
    goal_x = planner_states['x'][idx_goal]
    goal_y = planner_states['y'][idx_goal]
    goal_z = planner_states['z'][idx_goal]  
    goal_u = 25
    goal_v = 0
    goal_w = 0
    goal_phi = np.deg2rad(planner_states['phi_dg'][idx_goal])
    goal_theta = np.deg2rad(planner_states['theta_dg'][idx_goal])
    goal_psi = np.deg2rad(planner_states['psi_dg'][idx_goal])
    goal_p = 0
    goal_q = 0
    goal_r = 0

    start_state = np.array([init_x, init_y, init_z,
                            init_u, init_v, init_w,
                            init_phi, init_theta, init_psi,
                            init_p, init_q, init_r])
    
    goal_state = np.array([goal_x, goal_y, goal_z,
                            goal_u, goal_v, goal_w,
                            goal_phi, goal_theta, goal_psi,
                            goal_p, goal_q, goal_r])
    
    init_controls = np.array([init_al, init_el, init_rud, init_throttle])
    
    fw_mpc.initDecisionVariables()
    fw_mpc.reinitStartGoal(start_state, goal_state)
    fw_mpc.computeCost()
    fw_mpc.defineBoundaryConstraints()
    fw_mpc.addAdditionalConstraints()

    control_info, state_info = fw_mpc.solveMPCRealTimeStatic(start_state, goal_state,
                                                             init_controls)

    
    control_dict = fw_mpc.unpack_controls(control_info)
    state_dict = fw_mpc.unpack_states(state_info)

    
    ## simulator stuff
    init_states = {'x':init_x, 'y':init_y, 'z':init_z,
                     'u':init_u, 'v':init_v, 'w':init_w,
                     'phi':init_phi, 'theta':init_theta, 'psi':init_psi,
                     'p':init_p, 'q':init_q, 'r':init_r}
    

    aircraft_info = AircraftInfo(
        airplane_params,
        init_states,
        init_controls)
    
    aircraft_dynamics_rk = AircraftDynamics(aircraft_info)
    
    dt = mpc_params['dt_val']        
    t_init = 0.0
    t_final = 10

    tolerance = 5

    max_iter = 1000
    counter = 0

    rk_states = []
    rk_controls = []

    rest_waypoints = planner_states.iloc[idx_goal:,:]
    print("rest_waypoints: ", rest_waypoints)
    wp_max = len(rest_waypoints)
    
    for i, wp in enumerate(rest_waypoints.iterrows()):
        #compute distance error 
        current_states = aircraft_info.get_states()
        current_x = current_states[0]
        current_y = current_states[1]
        current_z = current_states[2]
        goal_x = wp[1]['x']
        goal_y = wp[1]['y']
        goal_z = wp[1]['z']        
        
        if i == 0:
            goal_theta = np.arctan2(wp[1]['z'] - \
                current_z, np.sqrt((wp[1]['x'] - current_x)**2 + (wp[1]['y'] - current_y)**2))
            
            goal_psi = np.arctan2(wp[1]['y'] - current_y,
                                  wp[1]['x'] - current_x)
        else:        
            prev_wp = rest_waypoints.iloc[i-1]
            goal_theta = np.arctan2(wp[1]['z'] - \
                prev_wp['z'], np.sqrt((wp[1]['x'] - prev_wp['x'])**2 + (wp[1]['y'] - prev_wp['y'])**2))
            goal_psi = np.arctan2(wp[1]['y'] - prev_wp['y'],
                                    wp[1]['x'] - prev_wp['x'])
            
        wp[1]['theta_dg'] = np.rad2deg(goal_theta)
        wp[1]['psi_dg']   = np.rad2deg(goal_psi)
        
        print("goal_theta", np.rad2deg(goal_theta))
        print("goal_psi", np.rad2deg(goal_psi))

        goal_states = np.array([goal_x, goal_y, goal_z,
                                goal_u, goal_v, goal_w,
                                goal_phi, goal_theta, goal_psi,
                                goal_p, goal_q, goal_r])
        

        if counter >= max_iter:
            break
        
        print("New goal location: ", goal_x, goal_y, goal_z)
        #print("wp percent", float(i/wp_max)*100)
        print("waypoint index", 1 + i) #added 1 since we start off the start

        error_x = goal_x - current_x
        error_y = goal_y - current_y
        error_z = goal_z - current_z
        
        error_mag = np.sqrt(error_x**2 + error_y**2 + error_z**2)

        while error_mag >= tolerance and counter <= max_iter:    

            if error_mag <= tolerance:
                print("Reached error_mag: ", error_mag)

            fw_mpc.reinitStartGoal(current_states, goal_state)

            #compute distance error 
            current_states = aircraft_info.get_states()
            current_x = current_states[0]
            current_y = current_states[1]
            current_z = current_states[2]

            error_x = goal_x - current_x
            error_y = goal_y - current_y
            error_z = goal_z - current_z            
        
            error_mag = np.sqrt(error_x**2 + error_y**2 + error_z**2)
            
            old_error_mag = error_mag

            control_info, state_info = fw_mpc.solveMPCRealTimeStatic(
                current_states, goal_state, init_controls)
            
            control_dict = fw_mpc.unpack_controls(control_info)
            state_dict = fw_mpc.unpack_states(state_info)

            #update the states with the        
            for i in range(len(control_dict['delta_e'])-1):
                input_aileron = control_dict['delta_a'][i]
                input_elevator = control_dict['delta_e'][i]
                input_rudder = control_dict['delta_r'][i]
                input_throttle = control_dict['delta_t'][i]
                
                x = state_dict['x'][i]
                y = state_dict['y'][i]
                z = state_dict['z'][i]
                u = state_dict['u'][i]
                v = state_dict['v'][i]
                w = state_dict['w'][i]
                phi = state_dict['phi'][i]
                theta = state_dict['theta'][i]
                psi = state_dict['psi'][i]
                p = state_dict['p'][i]
                q = state_dict['q'][i]
                r = state_dict['r'][i]

                new_states = np.array([x, y, z, 
                                    u, v, w, 
                                    phi, theta, psi, 
                                    p, q, r])

                #compute error between planner and simulator
                aircraft_info.update_states(new_states)
                rk_states.append(new_states)
                rk_controls.append([input_aileron, input_elevator, 
                                    input_rudder, input_throttle])

                init_x = new_states[0]
                init_y = new_states[1]
                init_z = new_states[2]
                init_u = new_states[3]
                init_v = new_states[4]
                init_w = new_states[5]
                init_phi = new_states[6]
                init_theta = new_states[7]
                init_psi = new_states[8]

                init_al = control_dict['delta_a'][i]
                init_el = control_dict['delta_e'][i]
                init_rud = control_dict['delta_r'][i]
                init_throttle = control_dict['delta_t'][i]
                
                start_state = np.array([init_x, init_y, init_z,
                                        init_u, init_v, init_w,
                                        init_phi, init_theta, init_psi,
                                        init_p, init_q, init_r])
                
                error_x = goal_x - new_states[0]
                error_y = goal_y - new_states[1]
                error_z = goal_z - new_states[2]         
            
                error_mag = np.sqrt(error_x**2 + error_y**2 + error_z**2)

                init_controls = np.array([init_al, 
                                        init_el, 
                                        init_rud, 
                                        init_throttle])

                # print("error_mag: ", error_mag)
                # print("difference: ", error_mag - old_error_mag)

                if error_mag <= tolerance:
                    print("Reached error_mag: ", error_mag)
                    break


                old_error_mag = error_mag


            counter += 1
            # print("counter: ", counter)

    #%%
    # plot in 2D
    plt.close('all')
    rk_states = pd.DataFrame(rk_states)
    rk_states.columns = ['x','y','z','u','v','w','phi','theta','psi','p','q','r']

    rk_controls = pd.DataFrame(rk_controls)
    rk_controls.columns = ['delta_a', 'delta_e', 'delta_r', 'delta_t']

    #plot in 2D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(rk_states['x'], rk_states['y'], rk_states['z'], '--', label='rk45', )
    #plot goal
    ax.plot([goal_x], [goal_y], [goal_z], 'o', label='goal')

    #plot all the waypoints
    ax.plot(planner_states['x'], planner_states['y'], planner_states['z'], 'o', label='waypoints')

    #loop through the waypoints and plot the heading
    for i, wp in planner_states.iterrows():
        ax.quiver3D(wp['x'], wp['y'], wp['z'],
                    np.cos(np.deg2rad(wp['psi_dg'])),
                    np.sin(np.deg2rad(wp['psi_dg'])),
                    np.sin(np.deg2rad(wp['theta_dg'])), length=10, color='k')

    # for wp, planner_states.iterrows:
    #     ax.quiver3D(wp[1]['x'], wp[1]['y'], wp[1]['z'],
    #                 np.cos(np.deg2rad(wp[1]['psi_dg'])),
    #                 np.sin(np.deg2rad(wp[1]['psi_dg'])),
    #                 0, length=5)
    
    #plot the state_dict
    ax.plot(state_dict['x'], 
            state_dict['y'], 
            state_dict['z'], '-.', label='mpc')
    ax.legend()

    #set z axis limits
    #ax.set_zlim(0, 30)
    
    #plot velocities in a subplot
    fig, ax = plt.subplots(3,1, sharex=True)
    ax[0].plot(rk_states['u'], label='u')
    ax[0].set_ylabel('u (m/s)')
    ax[0].legend()
    ax[1].plot(rk_states['v'], label='v')
    ax[1].set_ylabel('v (m/s)')
    ax[1].legend()
    ax[2].plot(rk_states['w'], label='w')
    ax[2].set_ylabel('w (m/s)')

    #plot euler angles in a subplot
    fig, ax = plt.subplots(3,1, sharex=True)
    ax[0].plot(np.rad2deg(rk_states['phi']), label='phi')
    ax[0].set_ylabel('phi (deg)')
    ax[0].legend()
    ax[1].plot(np.rad2deg(rk_states['theta']), label='theta')
    ax[1].set_ylabel('theta (deg)')
    ax[1].legend()
    ax[2].plot(np.rad2deg(rk_states['psi']), label='psi')
    
    #plot controls
    fig, ax = plt.subplots(4,1, sharex=True)
    ax[0].plot(np.rad2deg(rk_controls['delta_a']), label='delta_a')
    ax[0].set_ylabel('delta_a (deg)')
    ax[0].legend()
    ax[1].plot(np.rad2deg(rk_controls['delta_e']), label='delta_e')
    ax[1].set_ylabel('delta_e (deg)')
    ax[1].legend()
    ax[2].plot(np.rad2deg(rk_controls['delta_r']), label='delta_r')
    ax[2].set_ylabel('delta_r (deg)')
    ax[2].legend()
    ax[3].plot(rk_controls['delta_t'], label='delta_t')
    ax[3].set_ylabel('delta_t (N)')

    #visualize 
    data_vis = DataVisualization(rk_states, 5)
    ani = data_vis.animate_local(interval=20)
    ani_2 = data_vis.animate_global()    
    plt.show()
    