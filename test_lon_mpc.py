import numpy as np
import pandas as pd
import casadi as ca
import pickle as pkl
import matplotlib.pyplot as plt

from src.Utils import read_lon_matrices, get_airplane_params
from src.aircraft.AircraftDynamics import LonAirPlaneCasadi, LonAirPlane
from src.mpc.FixedWingMPC import LongitudinalMPC
from src.math_lib.VectorOperations import euler_dcm_body_to_inertial, euler_dcm_inertial_to_body, \
    compute_B_matrix

folder_dir = "src/aircraft/derivative_info/"
file_name = "example.txt"

full_path = folder_dir + file_name

# A_lon,B_lon = read_lon_matrices(full_path)
df = pd.read_csv("Coeffs.csv")
airplane_params = get_airplane_params(df)
lon_airplane = LonAirPlane(airplane_params)
u_0 = 20
theta_0 = 0
A_lon = lon_airplane.compute_A(u_0, theta_0)
B_lon = lon_airplane.compute_B(u_0)

#load the matrices
with open('A.pkl', 'rb') as f:
    A_lon = pkl.load(f)
    
with open('B.pkl', 'rb') as f:
    B_lon = pkl.load(f)

# lon_aircraft = LonAirPlaneCasadi(airplane_params, True,
#                                  A_lon, True, B_lon)

lon_aircraft = LonAirPlaneCasadi(airplane_params)
lon_aircraft.set_state_space()

Q = np.diag([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
R = np.diag([0.0, 0.0])

mpc_params = {
    'model': lon_aircraft,
    'dt_val': 0.05,
    'N': 10,
    'Q': Q,
    'R': R,
}

lon_mpc_constraints = {
    'delta_e_min': np.deg2rad(-30),
    'delta_e_max': np.deg2rad(30),
    'delta_t_min': 0.05,
    'delta_t_max': 0.75,
    'u_min': 20,
    'u_max': 35,
    'w_min': -10,
    'w_max': 10,
    'q_min': np.deg2rad(-60),
    'q_max': np.deg2rad(60),
    'theta_min': np.deg2rad(-35),
    'theta_max': np.deg2rad(35),
}

states = {
    'u': 25,
    'w': 0.0,
    'q': 0,
    'theta': np.deg2rad(-0.03),
    'h': 0.0,
    'x': 0.0,
}

controls = {
    'delta_e': np.deg2rad(0),
    'delta_t': 0.15,
}

#starting conditions -> wrap this to a function or something
start_state = np.array([states['u'], 
                        states['w'], 
                        states['q'], 
                        states['theta'],
                        states['h'], 
                        states['x']])


start_control = np.array([controls['delta_e'], 
                          controls['delta_t']])

#terminal conditions
goal_height = -20.0
goal_theta = -7.2 
goal_x = 250
goal_state = np.array([25, 
                       0, 
                       np.deg2rad(0), 
                       np.deg2rad(goal_theta),
                       goal_height, 
                       goal_x])

#begin mpc
lon_mpc = LongitudinalMPC(mpc_params, lon_mpc_constraints)

lon_mpc.initDecisionVariables()
lon_mpc.reinitStartGoal(start_state, goal_state)
lon_mpc.computeCost()
lon_mpc.defineBoundaryConstraints()
lon_mpc.addAdditionalConstraints()

control_results, state_results = lon_mpc.solveMPCRealTimeStatic(
    start_state, goal_state, start_control)

#unpack the results
control_results = lon_mpc.unpack_controls(control_results)
state_results = lon_mpc.unpack_states(state_results)

## simulate the trajectory of the aircraft
t_final = 5 #seconds
idx_start = 1

control_history = []
state_history = []
position_history = []
goal_history = []

x_original = 0
y_original = 0
z_original = 0

N = int(t_final/mpc_params['dt_val'])
time_current = 0

for i in range(N):

    if time_current > t_final/2:
        new_vel = 25
        new_theta = goal_theta
        new_height = 5.0
        new_x = goal_x
        goal_state = np.array([new_vel, 
                               0, 
                               0.0, 
                               np.deg2rad(new_theta),
                               new_height,
                               new_x])
        
    lon_mpc.reinitStartGoal(start_state, goal_state)

    control_results, state_results = lon_mpc.solveMPCRealTimeStatic(
        start_state, goal_state, start_control)
    
    #unpack the results
    control_results = lon_mpc.unpack_controls(control_results)
    state_results = lon_mpc.unpack_states(state_results)
    
    #update the initial states and controls
    start_state = np.array([state_results['u'][idx_start], 
                            state_results['w'][idx_start], 
                            state_results['q'][idx_start], 
                            state_results['theta'][idx_start],
                            state_results['h'][idx_start],
                            state_results['x'][idx_start]])
    
    start_control = np.array([control_results['delta_e'][idx_start],
                                control_results['delta_t'][idx_start]])
    
    
    #store the result in history
    state_history.append(start_state)
    control_history.append(start_control)
    
    R = euler_dcm_body_to_inertial(0, start_state[3], 0)
    body_vel = np.array([start_state[0], 0, start_state[1]])
    inertial_vel = np.matmul(R, body_vel)
    
    inertial_pos = inertial_vel * mpc_params['dt_val']
    print("inertial_pos: ", inertial_pos)
    inertial_pos = inertial_pos + np.array([x_original, y_original, z_original])
    
    #I was wrong this is the correct way to do it
    inertial_pos[0] = start_state[5]
    inertial_pos[2] = start_state[4]
    
    x_original = inertial_pos[0]
    y_original = inertial_pos[1]
    z_original = inertial_pos[2]
    
    position_history.append(inertial_pos)
    goal_history.append(goal_state)
    
    time_current += mpc_params['dt_val']
    
print("final position: ", inertial_pos)
# get global position of the aircraft
x_ned = [x[0] for x in position_history]
y_ned = [x[1] for x in position_history]
z_ned = [x[2] for x in position_history]

u = [x[0] for x in state_history]
w = [x[1] for x in state_history]
q = [x[2] for x in state_history]
theta = [x[3] for x in state_history]
h = [x[4] for x in state_history]

delta_t = [x[1] for x in control_history]
delta_e = [x[0] for x in control_history]

u_goal = [x[0] for x in goal_history]
w_goal = [x[1] for x in goal_history]
q_goal = [x[2] for x in goal_history]
theta_goal = [x[3] for x in goal_history]
h_goal = [x[4] for x in goal_history]

#compute angle of attack
alpha = np.arctan(np.array(w)/np.array(u))

#%% 
time_vec = np.arange(0, len(delta_t)*mpc_params['dt_val'], mpc_params['dt_val'])

# #create a line for the goal state
# u_goal = np.ones(len(time_vec)) * goal_state[0]
# w_goal = np.ones(len(time_vec)) * goal_state[1]
# q_goal = np.ones(len(time_vec)) * goal_state[2]
# theta_goal = np.ones(len(time_vec)) * goal_state[3]
# h_goal = np.ones(len(time_vec)) * goal_state[4]

#drop last element of time_vec
print("len time_vec: ", len(time_vec))
plt.close('all')
#plot the results
fig,ax = plt.subplots(6,1, figsize=(10,10))
ax[0].plot(time_vec,u, label='u')
ax[0].plot(time_vec,u_goal, label='u_goal', linestyle='--')
ax[0].set_ylabel('u (m/s)')

ax[1].plot(time_vec,w, label='w')
ax[1].plot(time_vec,w_goal, label='w_goal', linestyle='--')
ax[1].set_ylabel('w (m/s)')

ax[2].plot(time_vec,np.rad2deg(q), label='q')
ax[2].plot(time_vec,np.rad2deg(q_goal), label='q_goal', linestyle='--')
ax[2].set_ylabel('q (deg/s)')

ax[3].plot(time_vec,np.rad2deg(theta), label='theta')
ax[3].plot(time_vec,np.rad2deg(theta_goal), label='theta_goal', linestyle='--')
ax[3].set_ylabel('theta (deg)')

ax[4].plot(time_vec,h, label='h')
ax[4].plot(time_vec,h_goal, label='h_goal', linestyle='--')
ax[4].set_ylabel('h (meters)')

ax[5].plot(time_vec,np.rad2deg(alpha), label='alpha')
ax[5].set_ylabel('alpha (deg)')

#show legend
ax[0].legend()
ax[1].legend()
ax[2].legend()
ax[3].legend()
ax[4].legend()
ax[5].legend()


#%% 
#plot time_vec,controls
time_vec = time_vec
time_vec = np.arange(0, len(delta_e)*mpc_params['dt_val'], mpc_params['dt_val'])

fig,ax = plt.subplots(2,1, figsize=(10,10))
ax[0].plot(time_vec,np.rad2deg(delta_e), label='delta_e')
ax[0].set_ylabel('delta_e (deg)')
ax[1].plot(time_vec,delta_t, label='delta_t')
ax[1].set_ylabel('delta_t')

#plot time_vec,position in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
z_ned = -np.array(z_ned)

ax.plot(x_ned, y_ned, z_ned)
#show start and end point
ax.scatter(x_ned[0], y_ned[0], z_ned[0], marker='o', label='start')
ax.scatter(x_ned[-1], y_ned[-1], z_ned[-1], marker='x', label='end')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
#show legend
ax.legend()
ax.set_title('Position of Aircraft in 3D in NEU Frame')

plt.show() 

# N = 50

# for i in range(N):
    
#     u = lon_mpc.solve(init_states, init_controls)
#     u_dict = lon_mpc.unpack_controls(u)
#     x_dict = lon_mpc.unpack_states(lon_mpc.x0)
    
#     init_states['u'] = x_dict['u'][1]
#     init_states['w'] = x_dict['w'][1]
#     init_states['q'] = x_dict['q'][1]
#     init_states['theta'] = x_dict['theta'][1]
#     init_states['h'] = x_dict['h'][1]
    
#     init_controls['delta_e'] = u_dict['delta_e'][1]
#     init_controls['delta_t'] = u_dict['delta_t'][1]
    
#     lon_mpc.update_x0(init_states)
#     lon_mpc.update_u0(init_controls)
    
#     print("u: ", init_states['u'])
#     print("w: ", init_states['w'])
#     print("q: ", init_states['q'])
#     print("theta: ", init_states['theta'])
#     print("h: ", init_states['h'])
#     print("delta_e: ", init_controls['delta_e'])
#     print("delta_t: ", init_controls['delta_t'])
#     print("")
    
    

