"""
Testing the linearized MPC and feeding it back to 6DOF simulation
- Use 6dof states to close the loop and send it back to MPC 

"""

import numpy as np
import pandas as pd
import casadi as ca
import pickle as pkl
import matplotlib.pyplot as plt

from src.Utils import read_lon_matrices, get_airplane_params

from src.aircraft.AircraftDynamics import LinearizedAircraft, \
    LinearizedAircraftCasadi, AircraftDynamics     
from src.aircraft.Aircraft import AircraftInfo
from src.mpc.FixedWingMPC import LinearizedAircraftMPC

from src.math_lib.VectorOperations import euler_dcm_body_to_inertial, euler_dcm_inertial_to_body, \
    compute_B_matrix
    
from src.aircraft.AircraftDynamics import convert_lin_controls_to_regular,\
    convert_lin_states_to_regular, convert_lin_states_to_regular, \
    convert_lin_controls_to_regular, convert_regular_states_to_lin, \
    convert_regular_controls_to_lin

    
df = pd.read_csv("SIM_Plane_h_vals.csv")
airplane_params = get_airplane_params(df)

with open('A_full.pkl', 'rb') as f:
    A_full = pkl.load(f)
    
with open('B_full.pkl', 'rb') as f:
    B_full = pkl.load(f)
    
lin_aircraft_ca = LinearizedAircraftCasadi(airplane_params, 
                                           A_full, 
                                           B_full)

#terminal conditions
goal_u = 15.0
goal_w = 0.0
goal_q = 0.0
goal_theta = np.deg2rad(-0.03)
goal_h = 5.0
goal_x = 250
goal_v = 0.0
goal_p = 0.0
goal_r = 0.0
goal_phi = np.deg2rad(45.0)
goal_psi = np.deg2rad(30.0)
goal_y = 0.0

#weighting matrices for state
Q = np.diag([
    1.0, #u
    0.0, #w
    0.0, #q
    0.0, #theta
    1.0, #h
    0.0, #x
    0.0, #v
    0.0, #p
    0.0, #r
    0.0, #phi
    0.0, #psi
    0.0, #y
])

R = np.diag([
    1.0, #delta_e
    1.0, #delta_t
    1.0, #delta_a
    1.0, #delta_r
])

mpc_params = {
    'model': lin_aircraft_ca,
    'dt_val': 0.01,
    'N': 10,
    'Q': Q,
    'R': R,
}

lin_mpc_constraints = {
    'delta_e_min': np.deg2rad(-30),
    'delta_e_max': np.deg2rad(30),
    'delta_t_min': 0.05,
    'delta_t_max': 0.75,
    'delta_a_min': np.deg2rad(-25),
    'delta_a_max': np.deg2rad(25),
    'delta_r_min': np.deg2rad(-30),
    'delta_r_max': np.deg2rad(30),
    'u_min': 15,
    'u_max': 35,
    'w_min': -0.5,
    'w_max': 0.5,
    'q_min': np.deg2rad(-60),
    'q_max': np.deg2rad(60),
    'theta_min': np.deg2rad(-35),
    'theta_max': np.deg2rad(35),
    'v_min': -35, #don't need this really
    'v_max': 35,  #don't need this really
    'p_min': np.deg2rad(-60),
    'p_max': np.deg2rad(60),
    'r_min': np.deg2rad(-60),
    'r_max': np.deg2rad(60),
    'phi_min': np.deg2rad(-60),
    'phi_max': np.deg2rad(60)
}

states = {
    'u': 25,
    'w': 0.0,
    'q': 0,
    'theta': np.deg2rad(-0.03),
    'h': 0.0,
    'x': 0.0,
    'v': 0.0,
    'p': 0.0,
    'r': 0.0,
    'phi': 0.0,
    'psi': 0.0,
    'y': 0.0,
}

controls = {
    'delta_e': np.deg2rad(0),
    'delta_t': 0.1,
    'delta_a': np.deg2rad(0),
    'delta_r': np.deg2rad(0),
}

start_state = np.array([states['u'],
                        states['w'],
                        states['q'],
                        states['theta'],
                        states['h'],
                        states['x'],
                        states['v'],
                        states['p'],
                        states['r'],
                        states['phi'],
                        states['psi'],
                        states['y']])

start_control = np.array([controls['delta_e'],
                          controls['delta_t'],
                          controls['delta_a'],
                          controls['delta_r']])


goal_state = np.array([goal_u,
                          goal_w,
                          goal_q,
                          goal_theta,
                          goal_h,
                          goal_x,
                          goal_v,
                          goal_p,
                          goal_r,
                          goal_phi,
                          goal_psi,
                          goal_y])

#begin mpc
lin_mpc = LinearizedAircraftMPC(mpc_params, lin_mpc_constraints)

lin_mpc.initDecisionVariables()
lin_mpc.reinitStartGoal(start_state, goal_state)
lin_mpc.computeCost()
lin_mpc.defineBoundaryConstraints()
lin_mpc.addAdditionalConstraints()

control_results, state_results = lin_mpc.solveMPCRealTimeStatic(
    start_state, goal_state, start_control)

#unpack the results
control_results = lin_mpc.unpack_controls(control_results)
state_results = lin_mpc.unpack_states(state_results)

## simulate the trajectory of the aircraft
t_final = 3 #seconds
idx_start = 1

control_history = []
state_history = []
position_history = []
state_position_history = []
goal_history = []

# initialize the simulator
aircraft_info = AircraftInfo(airplane_params, states, controls)
aircraft_dynamics = AircraftDynamics(aircraft_info)

x_original = 0
y_original = 0
z_original = 0

N = int(t_final/mpc_params['dt_val'])
time_current = 0

for i in range(N):
    
    # if time_current > t_final/2:
    new_vel = 20.0
    new_w = 0.0
    new_q = 0.0
    new_theta = np.deg2rad(-0.03)
    new_height = 0.0
    new_x = 250
    new_v = 0.0
    new_p = 0.0     
    new_r = 0.0
    new_phi = np.deg2rad(45.0) 
    new_psi = np.deg2rad(5.0) #+ state_results['psi'][idx_start]
    new_y = 0.0
    goal_state = np.array([new_vel,
                                new_w,
                                new_q,
                                new_theta,
                                new_height,
                                new_x,
                                new_v,
                                new_p,
                                new_r,
                                new_phi,
                                new_psi,
                                new_y])
        
        
    lin_mpc.reinitStartGoal(start_state, goal_state)
    
    control_results, state_results = lin_mpc.solveMPCRealTimeStatic(
        start_state, goal_state, start_control)    
    #unpack the results
    control_results = lin_mpc.unpack_controls(control_results)
    state_results = lin_mpc.unpack_states(state_results)
    
    ## this is where I need to do a step input for the 6DOF simulation
    start_state = np.array([state_results['u'][idx_start],
                              state_results['w'][idx_start],
                              state_results['q'][idx_start],
                              state_results['theta'][idx_start],
                              state_results['h'][idx_start],
                              state_results['x'][idx_start],
                              state_results['v'][idx_start],
                              state_results['p'][idx_start],
                              state_results['r'][idx_start],
                              state_results['phi'][idx_start],
                              state_results['psi'][idx_start],
                              state_results['y'][idx_start]])
    
    start_control = np.array([control_results['delta_e'][idx_start],
                            control_results['delta_t'][idx_start],
                            control_results['delta_a'][idx_start],
                            control_results['delta_r'][idx_start]])
        
    input_aileron  = control_results['delta_a'][idx_start]
    input_rudder   = control_results['delta_r'][idx_start]
    input_elevator = control_results['delta_e'][idx_start]
    input_throttle = control_results['delta_t'][idx_start]
    
    sim_states = convert_lin_states_to_regular(start_state)
    
    actual_states = aircraft_dynamics.rk45(input_aileron, 
                                            input_elevator,
                                            input_rudder, 
                                            input_throttle,
                                            sim_states,                                    
                                            mpc_params['dt_val'])

    start_state = convert_regular_states_to_lin(actual_states)

    #store the result in history
    control_history.append(start_control)
    state_history.append(start_state)
    
    # R = euler_dcm_body_to_inertial(state_results['phi'][idx_start],
    #                                state_results['theta'][idx_start],
    #                                state_results['psi'][idx_start])
    
    # body_vel = np.array([state_results['u'][idx_start],
    #                     state_results['v'][idx_start],
    #                     state_results['w'][idx_start]])
    
    # inertial_vel = np.matmul(R, body_vel)
    # inertial_pos = inertial_vel * mpc_params['dt_val']
    
    # x_original = x_original + inertial_pos[0]
    # y_original = y_original + inertial_pos[1]
    # z_original = z_original + inertial_pos[2]
    # position_history.append(np.array([x_original, y_original, z_original]))    

    # state_position_history.append(np.array([state_results['x'][idx_start],
    #                                         state_results['y'][idx_start],
    #                                         state_results['h'][idx_start]]))
    
    
    # #replace the position with the inertial position
    # start_state[5] = x_original
    # start_state[11] = y_original
    # start_state[4] = z_original
    
    # inertial_position = np.array([state_results['x'][idx_start],
    #                                 state_results['y'][idx_start],
    #                                 state_results['h'][idx_start]])
    
    #position_history.append(inertial_position)
    goal_history.append(goal_state)
    
    time_current += mpc_params['dt_val']


u = [x[0] for x in state_history]
w = [x[1] for x in state_history]
q = [x[2] for x in state_history]
theta = [x[3] for x in state_history]
h = [x[4] for x in state_history]
x = [x[5] for x in state_history]
v = [x[6] for x in state_history]
p = [x[7] for x in state_history]
r = [x[8] for x in state_history]
phi = [x[9] for x in state_history]
psi = [x[10] for x in state_history]
y = [x[11] for x in state_history]

delta_e = [x[0] for x in control_history]
delta_t = [x[1] for x in control_history]
delta_a = [x[2] for x in control_history]
delta_r = [x[3] for x in control_history]

#save controls to csv
df = pd.DataFrame({'delta_e': delta_e,
                    'delta_t': delta_t,
                    'delta_a': delta_a,
                    'delta_r': delta_r})

df.to_csv('MPC_Plane_controls.csv', index=False)

#save states to csv
states_df = pd.DataFrame({'u': u,
                            'w': w,
                            'q': q,
                            'theta': theta,
                            'h': h,
                            'x': x,
                            'v': v,
                            'p': p,
                            'r': r,
                            'phi': phi,
                            'psi': psi,
                            'y': y})

states_df.to_csv('MPC_Plane_states.csv', index=False)


u_goal = [x[0] for x in goal_history]
w_goal = [x[1] for x in goal_history]
q_goal = [x[2] for x in goal_history]
theta_goal = [x[3] for x in goal_history]
h_goal = [x[4] for x in goal_history]
x_goal = [x[5] for x in goal_history]
v_goal = [x[6] for x in goal_history]
p_goal = [x[7] for x in goal_history]
r_goal = [x[8] for x in goal_history]
phi_goal = [x[9] for x in goal_history]
psi_goal = [x[10] for x in goal_history]
y_goal = [x[11] for x in goal_history]

#compute angle of attack flipped w since alpha is positive when pitching up
alpha = np.arctan(-np.array(w)/np.array(u))

#%%
plt.close('all')
time_vec = np.linspace(0, t_final, N)
fig,ax = plt.subplots(6,2)
ax[0,0].plot(time_vec, u, label='u')
ax[0,0].plot(time_vec, u_goal, label='u_goal', linestyle='--')
ax[0,0].set_ylabel("u (m/s)")

ax[1,0].plot(time_vec, w, label='w')
ax[1,0].plot(time_vec, w_goal, linestyle='--')
ax[1,0].set_ylabel("w (m/s)")

ax[2,0].plot(time_vec, np.rad2deg(q), label='q')
ax[2,0].plot(time_vec, np.rad2deg(q_goal), label='q_goal', linestyle='--')
ax[2,0].set_ylabel("q (deg/s)")

ax[3,0].plot(time_vec, np.rad2deg(theta))
ax[3,0].plot(time_vec, np.rad2deg(theta_goal), label='goal', linestyle='--')
ax[3,0].set_ylabel("theta (deg)")

ax[4,0].plot(time_vec, h)
ax[4,0].plot(time_vec, h_goal, label='goal', linestyle='--')
ax[4,0].set_ylabel("h (m)")

ax[5,0].plot(time_vec, x)
ax[5,0].set_ylabel("x (m)")

#have plots share x axis
for i in range(6):
    ax[i,0].sharex(ax[i,1])
    ax[i,0].legend()
    
ax[0,1].plot(time_vec, v)
ax[0,1].set_ylabel("v (m/s)")

ax[1,1].plot(time_vec, np.rad2deg(p))
ax[1,1].set_ylabel("p (deg/s)")

ax[2,1].plot(time_vec, np.rad2deg(r))
ax[2,1].set_ylabel("r (deg/s)")

ax[3,1].plot(time_vec, np.rad2deg(phi))
ax[3,1].plot(time_vec, np.rad2deg(phi_goal), label='goal', linestyle='--')
ax[3,1].set_ylabel("phi (deg)")


ax[4,1].plot(time_vec, np.rad2deg(psi))
ax[4,1].plot(time_vec, np.rad2deg(psi_goal), label='goal', linestyle='--')
ax[4,1].set_ylabel("psi (deg)")


ax[5,1].plot(time_vec, y)
ax[5,1].set_ylabel("y (m)")

for i in range(6):
    ax[i,1].sharex(ax[i,0])
    ax[i,1].legend()


#plot control history
fig,ax = plt.subplots(4,1)
ax[0].plot(time_vec, np.rad2deg(delta_e))
ax[0].set_ylabel("delta_e (deg)")
ax[1].plot(time_vec, delta_t)
ax[1].set_ylabel("delta_t")
ax[2].plot(time_vec, np.rad2deg(delta_a))
ax[2].set_ylabel("delta_a (deg)")
ax[3].plot(time_vec, np.rad2deg(delta_r))
ax[3].set_ylabel("delta_r (deg)")

#plot 3d trajectory
position_history = np.array(state_position_history)
z_ned = -np.array(h)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.plot(position_history[:,0], position_history[:,1], -position_history[:,2], 
#         label='trajectory (linearized)')
ax.plot(x, y, z_ned, label='rotation (actual position)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.scatter(x[0], y[0], z_ned[0], marker='o', label='start')
ax.scatter(x[-1], y[-1], z_ned[-1], marker='x', label='end')
ax.legend()
ax.set_title('3D Trajectory in NEU Frame')


airspeed = np.sqrt(np.array(u)**2 + +np.array(v)**2 + np.array(w)**2)

#plot airspeed
fig,ax = plt.subplots(1,1)
ax.plot(time_vec, airspeed)
ax.set_ylabel('airspeed (m/s)')
ax.set_xlabel('time (s)')

plt.show()