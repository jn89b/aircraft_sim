import numpy as np
import pandas as pd
import casadi as ca

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from src.mpc.FixedWingMPC import FixedWingMPC
from src.aircraft.AircraftDynamics import AircraftCasadi
from src.Utils import get_airplane_params

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
        'delta_t_max': 200.0, #newtons
        'delta_t_min': 5, #newtons
        'u_max': 30.0, #m/s
        'u_min': 15.0, #m/s
        'v_max': 5, #m/s
        'v_min': -5, #m/s
        'w_max': 10, #m/s
        'w_min': -10, #m/s
        'phi_max': np.deg2rad(45), #rad
        'phi_min': np.deg2rad(-45), #rad
        'theta_max': np.deg2rad(45), #rad
        'theta_min': np.deg2rad(-45), #rad
        # 'psi_max': np.deg2rad(45), #rad
        # 'psi_min': np.deg2rad(-45), #rad
        'p_max': np.deg2rad(np.inf), #rad/s
        'p_min': np.deg2rad(-np.inf), #rad/s
        'q_max': np.deg2rad(np.inf), #rad/s
        'q_min': np.deg2rad(-np.inf), #rad/s
        'r_max': np.deg2rad(np.inf), #rad/s
        'r_min': np.deg2rad(-np.inf), #rad/s
    }

    #create a diagonal matrix for the weights

    Q = ca.diag([1,1,1, #position
                 0,0,0, #velocity body
                 0,0,0.5, #euler angles
                 0,0,0]) #angular rates body
    print()

    R = ca.diag([0.5,
                 0.5,
                 0.5,
                 0.5]) #control inputs

    mpc_params = {
        'model': aircraft_ca,
        'dt_val': 0.01,
        'N': 30,
        'Q': Q,
        'R': R
    }

    fw_mpc = FixedWingMPC(mpc_params, mpc_constraints)
    
    #set initial conditions
    init_x = 10
    init_y = 60
    init_z = 5
    init_u = 25
    init_v = 0
    init_w = 0
    init_phi = 0
    init_theta = np.deg2rad(-5)
    init_psi = np.deg2rad(38)
    init_p = 0
    init_q = 0
    init_r = 0

    init_al = np.deg2rad(0)
    init_el = np.deg2rad(0)
    init_rud = 0
    init_throttle = 30

    goal_x = 15
    goal_y = 65
    goal_z = 6  
    goal_u = 25
    goal_v = 0
    goal_w = 0
    goal_phi = np.deg2rad(-21)
    goal_theta = np.deg2rad(0)
    goal_psi = np.deg2rad(45)
    goal_p = 0
    goal_q = 0
    goal_r = 0

    #load planner states
    planner_states = pd.read_csv("planner_states.csv")
    
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
    
    # new_controls ={}
    # for k,v in control_dict.items():
    #     print(k,v)
    #     new_controls[k] = v[1]

    aircraft_info = AircraftInfo(
        airplane_params,
        init_states,
        init_controls)
    
    aircraft_dynamics_rk = AircraftDynamics(aircraft_info)
    
    rk_states = []

    dt = 0.01        
    t_init = 0.0
    t_final = 10

    for i in range(len(control_dict['delta_e'])):
        input_aileron = control_dict['delta_a'][i]
        input_elevator = control_dict['delta_e'][i]
        input_rudder = control_dict['delta_r'][i]
        input_throttle = control_dict['delta_t'][i]
        new_states = aircraft_dynamics_rk.rk45(
            input_aileron,
            input_elevator,
            input_rudder,
            input_throttle,
            aircraft_info.states,
            dt
        )

        aircraft_info.update_states(new_states)
        rk_states.append(new_states)


    #%%
    # plot in 2D
    rk_states = pd.DataFrame(rk_states)
    rk_states.columns = ['x','y','z','u','v','w','phi','theta','psi','p','q','r']

    #plot in 2D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(rk_states['x'], rk_states['y'], rk_states['z'], '--', label='rk45', )
    #plot goal
    ax.plot([goal_x], [goal_y], [goal_z], 'o', label='goal')

    #plot the state_dict
    ax.plot(state_dict['x'], 
            state_dict['y'], 
            state_dict['z'], '-.', label='mpc')
    ax.legend()

    #plot velocities in a subplot
    fig, ax = plt.subplots(3,1, sharex=True)
    ax[0].plot(state_dict['u'], label='u')
    ax[0].set_ylabel('u (m/s)')
    ax[0].legend()
    ax[1].plot(state_dict['v'], label='v')
    ax[1].set_ylabel('v (m/s)')
    ax[1].legend()
    ax[2].plot(state_dict['w'], label='w')
    ax[2].set_ylabel('w (m/s)')

    plt.show()