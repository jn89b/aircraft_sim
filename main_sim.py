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


"""
Have MPC compute the trajectory
Feed it to the simulator 
- Return the actual value  


"""
if __name__ == "__main__":
    
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
        'delta_t_max': 300, #newtons
        'delta_t_min': 15, #newtons
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
    Q = ca.diag([1.0,1.0,0.75, #position
                 0.0,0.0,0.0, #velocity body
                 5.0,2.5,5.0, #euler angles
                 0,0,0]) #angular rates body

    R = ca.diag([2.0,
                 2.0,
                 2.0,
                 2.0]) #control inputs

    mpc_freq = 50
    mpc_params = {
        'model': aircraft_ca,
        'dt_val': 1/mpc_freq,
        'N': 10,
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
    
    print("initial heading" , np.rad2deg(init_psi))    
    init_theta = -np.arctan2(planner_states['z'][1] - planner_states['z'][0],
                            np.sqrt((planner_states['x'][1] - planner_states['x'][0])**2 + \
                            planner_states['y'][1] - planner_states['y'][0])**2)
    
    # init_theta = np.deg2rad(20) 
    init_p = 0
    init_q = 0
    init_r = 0

    init_al = np.deg2rad(0)
    init_el = np.deg2rad(0)
    init_rud = 0
    init_throttle = 40

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
    
    sim_dt = 1/500 # 1/frequency
    
    N = 1000
    
    control_idx = 0
    
    states_history = []
    
    #initialize the new states
    new_states_rk = init_states
    
    #tolerance 
    tolerance = 1.0
    
    for i in range(N):
        
        input_aileron = control_dict['delta_a'][control_idx]
        input_elevator = control_dict['delta_e'][control_idx]
        input_rudder = control_dict['delta_r'][control_idx]
        input_throttle = control_dict['delta_t'][control_idx]
        
        new_states_rk = aircraft_dynamics_rk.rk45(
            input_aileron,
            input_elevator,
            input_rudder,
            input_throttle,
            aircraft_info.states,
            sim_dt
        )
        

        start_state = new_states_rk
        aircraft_info.update_states(new_states_rk)
        states_history.append(new_states_rk)

        error_x = goal_x - new_states_rk[0]
        error_y = goal_y - new_states_rk[1]
        error_z = goal_z - new_states_rk[2]         

        error_mag = np.sqrt(error_x**2 + error_y**2 + error_z**2)

        if error_mag < tolerance:
            print("reached goal", new_states_rk)
            break
        
        if i // mpc_freq == 0:
            # print("recomputing mpc", i)
            print("error", error_mag)
            control_info, state_info = fw_mpc.solveMPCRealTimeStatic(
                start_state, goal_state,init_controls)
            control_dict = fw_mpc.unpack_controls(control_info)
            state_dict = fw_mpc.unpack_states(state_info)

        
    #set as pd 
    states = pd.DataFrame(states_history)
    states.columns = ['x', 'y', 'z', 'u', 'v', 'w', 'phi', 'theta', 'psi', 'p', 'q', 'r']
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':'3d'})
    ax.plot(states['x'], states['y'], states['z'], 'o-', 
            label='Trajectory',)
    
    #plot the goal
    ax.plot(planner_states['x'], planner_states['y'], planner_states['z'], 'o-',
            label='Waypoint Trajectory')
    # ax.plot(rk_states['x'], rk_states['y'], -rk_states['z'], 'x-' ,
    #         label='RK45')
    ax.legend()
    #plot attitudes in euler angles in a subplot
    fig1, ax1 = plt.subplots(3,1,sharex=True)
    ax1[0].plot(np.rad2deg(states['phi']), label='RK45')
    ax1[1].plot(np.rad2deg(states['theta']), label='RK45')
    ax1[2].plot(np.rad2deg(states['psi']), label='RK45')
    ax1[2].set_xlabel('Time (s)')
    ax1[0].set_ylabel('Roll (deg)')
    ax1[1].set_ylabel('Pitch (deg)')
    ax1[2].set_ylabel('Yaw (deg)')
    
    plt.show()

    
        
    
    
    
    