import numpy as np
import pandas as pd
import casadi as ca
import pickle as pkl
import matplotlib.pyplot as plt
import math as m

from src.Utils import read_lon_matrices, get_airplane_params
from src.aircraft.AircraftDynamics import LinearizedAircraft, LinearizedAircraftCasadi
from src.mpc.FixedWingMPC import LinearizedAircraftMPC
from src.math_lib.VectorOperations import euler_dcm_body_to_inertial, euler_dcm_inertial_to_body, \
    compute_B_matrix
    
from src.aircraft.AircraftDynamics import add_noise_to_linear_states
plt.close('all')

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
goal_h = -20.0
goal_x = 120
goal_v = 0.0
goal_p = 0.0
goal_r = 0.0
goal_phi = np.deg2rad(0.0)
goal_psi = np.deg2rad(30.0)
goal_y = -110


#weighting matrices for state
Q = np.diag([
    0.75, #u
    0.0, #w
    0.0, #q
    0.0, #theta
    1.0, #h
    0.0, #x
    0.0, #v
    0.0, #p
    0.0, #r
    0.0, #phi
    1.0, #psi
    0.0, #y
])

r_control = 0.4
R = np.diag([
    r_control, #delta_e
    1.0, #delta_t
    r_control, #delta_a
    r_control, #delta_r
])

mpc_params = {
    'model': lin_aircraft_ca,
    'dt_val': 0.05,
    'N': 15,
    'Q': Q,
    'R': R,
}

lin_mpc_constraints = {
    'delta_e_min': np.deg2rad(-30),
    'delta_e_max': np.deg2rad(30),
    'delta_t_min': 0.1,
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
    'phi_min': np.deg2rad(-45),
    'phi_max': np.deg2rad(45)
}

states = {
    'u': 15,
    'w': 0.0,
    'q': 0,
    'theta': np.deg2rad(-0.03),
    'h': -3.0,
    'x': 0.0,
    'v': 0.0,
    'p': 0.0,
    'r': 0.0,
    'phi': 0.0,
    'psi': np.deg2rad(45.0),
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


#load planner states
planner_states = pd.read_csv("planner_states.csv")

#recompute the planner states heading, I made a mistake in the planner
#The heading should be the heading to the next waypoint  
for i, wp in planner_states.iterrows():
    if i == 0:
        continue
    else:
        dx = wp['x'] - planner_states.iloc[i-1]['x']
        dy = wp['y'] - planner_states.iloc[i-1]['y']
        planner_states.iloc[i-1]['psi_dg'] = np.rad2deg(np.arctan2(dy,dx))
        # planner_states.iloc[i] = wp

idx_goal = 1

#load up planner states
goal_x = planner_states['x'][idx_goal]
goal_y = planner_states['y'][idx_goal]
goal_z = planner_states['z'][idx_goal]  
goal_u = 25
goal_v = 0
goal_w = 0
# goal_phi = np.deg2rad(planner_states['phi_dg'][idx_goal])
# goal_theta = np.deg2rad(planner_states['theta_dg'][idx_goal])
# goal_psi = np.deg2rad(planner_states['psi_dg'][idx_goal])
goal_p = 0
goal_q = 0
goal_r = 0


goal_state = np.array([goal_u,
                          goal_w,
                          goal_q,
                          goal_theta,
                          goal_z,
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
t_final = 10 #seconds
idx_start = 1

control_history = []
state_history = []
position_history = []
state_position_history = []
goal_history = []

x_original = 0
y_original = 0
z_original = 0

N = int(t_final/mpc_params['dt_val'])
time_current = 0

add_noise = False

#when aircraft approaches this tolerance, set to turnpoint heading
approach_tolerance = 16.0
tolerance = 8.0

rest_waypoints = planner_states.iloc[idx_goal:,:]
print("rest_waypoints: ", rest_waypoints)
wp_max = len(rest_waypoints)

FINISHED = False
for i in range(N):
    
    if FINISHED == True:
        print("reached goal")
        break
    
    for j, wp in enumerate(rest_waypoints.iterrows()):
        print("j", j)
        goal_x = wp[1]['x']
        goal_y = wp[1]['y']
        goal_h = wp[1]['z']        
        dy = goal_y - start_state[11]
        dx = goal_x - start_state[5]
        dz = goal_h - start_state[4]
        new_vel = 25.0
        new_w = 0.0
        new_q = 0.0
        new_theta = np.arctan2(-dz, np.sqrt(dx**2 + dy**2))
        new_height = goal_h
        new_x = 250
        new_v = 0.0
        new_p = 0.0     
        new_r = 0.0
        new_phi = np.deg2rad(-55.0)
        new_psi = np.arctan2(dy,dx)
        new_y = 0.0
        goal_state = np.array([new_vel,
                                new_w,
                                new_q,
                                new_theta,
                                goal_h,
                                goal_x,
                                new_v,
                                new_p,
                                new_r,
                                new_phi,
                                new_psi,
                                goal_y])

        error = np.sqrt(dx**2 + dy**2 + dz**2)
        
        while error >= tolerance:
            goal_x = wp[1]['x']
            goal_y = wp[1]['y']
            goal_h = wp[1]['z']        
            dy = goal_y - start_state[11]
            dx = goal_x - start_state[5]
            dz = goal_h - start_state[4]
            new_vel = 15.0
            new_w = 0.0
            new_q = 0.0
            new_theta = np.arctan2(-dz, np.sqrt(dx**2 + dy**2))
            new_height = goal_h
            new_x = 250
            new_v = 0.0
            new_p = 0.0     
            new_r = 0.0
            new_phi = np.deg2rad(0.0)
            new_y = 0.0
            
            if error >= approach_tolerance:
                new_psi = np.arctan2(dy,dx)
            else:
                new_psi = np.deg2rad(wp[1]['psi_dg'])
                
            goal_state = np.array([new_vel,
                                    new_w,
                                    new_q,
                                    new_theta,
                                    goal_h,
                                    goal_x,
                                    new_v,
                                    new_p,
                                    new_r,
                                    new_phi,
                                    new_psi,
                                    goal_y])
        
            lin_mpc.reinitStartGoal(start_state, goal_state)
            
            control_results, state_results = lin_mpc.solveMPCRealTimeStatic(
                start_state, goal_state, start_control)
            
            #unpack the results
            control_results = lin_mpc.unpack_controls(control_results)
            state_results = lin_mpc.unpack_states(state_results)
            
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
                
            #store the result in history
            control_history.append(start_control)
            state_history.append(start_state)
            
            R = euler_dcm_body_to_inertial(state_results['phi'][idx_start],
                                        state_results['theta'][idx_start],
                                        state_results['psi'][idx_start])
            
            body_vel = np.array([state_results['u'][idx_start],
                                state_results['v'][idx_start],
                                state_results['w'][idx_start]])
            
            inertial_vel = np.matmul(R, body_vel)
            inertial_pos = inertial_vel * mpc_params['dt_val']
            
            x_original = x_original + inertial_pos[0]
            y_original = y_original + inertial_pos[1]
            z_original = z_original + inertial_pos[2]
            position_history.append(np.array([x_original, y_original, z_original]))    

            state_position_history.append(np.array([state_results['x'][idx_start],
                                                    state_results['y'][idx_start],
                                                    state_results['h'][idx_start]]))
            

            #replace the position with the inertial position
            start_state[5] = x_original
            start_state[11] = y_original
            start_state[4] = z_original

            if add_noise == True:
                start_state = add_noise_to_linear_states(start_state)
                
            #position_history.append(inertial_position)
            goal_history.append(goal_state)
            
            time_current += mpc_params['dt_val']
            
            dy = goal_y - start_state[11]
            dx = goal_x - start_state[5]
            dz = goal_h - start_state[4]
            error = np.sqrt(dx**2 + dy**2 + dz**2)
            # if error <= tolerance:
            print("error: ", error)
            
        if j == wp_max - idx_start:
            FINISHED = True
            break

#%% 
x_ned = [x[0] for x in position_history]
y_ned = [x[1] for x in position_history]
z_ned = [x[2] for x in position_history]

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

time_vec = np.arange(0, len(u_goal))

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
z_ned = -np.array(z_ned)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(position_history[:,0], position_history[:,1], -position_history[:,2], 
        label='trajectory (linearized)')
ax.plot(x_ned, y_ned, z_ned, label='rotation (actual position)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.scatter(x_ned[0], y_ned[0], z_ned[0], marker='o', label='start')
ax.scatter(x_ned[-1], y_ned[-1], z_ned[-1], marker='x', label='end')
ax.scatter(goal_x, goal_y, -goal_h, marker='o', label='goal position')

#loop through the waypoints and plot the heading
for i, wp in planner_states.iterrows():
    ax.quiver3D(wp['x'], wp['y'], -wp['z'],
                np.cos(np.deg2rad(wp['psi_dg'])),
                np.sin(np.deg2rad(wp['psi_dg'])),
                np.sin(np.deg2rad(wp['theta_dg'])), length=20, color='k')


ax.legend()
ax.set_title('3D Trajectory in NEU Frame')


airspeed = np.sqrt(np.array(u)**2 + +np.array(v)**2 + np.array(w)**2)

#plot airspeed
fig,ax = plt.subplots(1,1)
ax.plot(time_vec, airspeed)
ax.set_ylabel('airspeed (m/s)')
ax.set_xlabel('time (s)')

plt.show()