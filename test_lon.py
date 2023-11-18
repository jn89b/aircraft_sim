import numpy as np
import pandas as pd
import casadi as ca

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

print("at u_0 = {} and theta_0 = {}".format(u_0, theta_0))

v_trim = u_0
lon_aircraft = LonAirPlaneCasadi(airplane_params)
lon_aircraft.set_state_space()

Q = np.diag([0, 1, 100, 100])
R = np.diag([0.1, 0.1])

mpc_params = {
    'model': lon_aircraft,
    'dt_val': 0.01,
    'N': 100,
    'Q': Q,
    'R': R,
}

lon_mpc_constraints = {
    'delta_e_min': -np.deg2rad(25),
    'delta_e_max': np.deg2rad(25),
    'delta_t_min': 0.1,
    'delta_t_max': 1,
    'u_min': 15,
    'u_max': 30,
    'w_min': -10,
    'w_max': 10,
    'q_min': np.deg2rad(-30),
    'q_max': np.deg2rad(30),
    'theta_min': np.deg2rad(-30),
    'theta_max': np.deg2rad(30),
}


states = {
    'u': 25,
    'w': 0.0,
    'q': 0,
    'theta': np.deg2rad(0),
}

controls = {
    'delta_e': np.deg2rad(0),
    'delta_t': lon_mpc_constraints['delta_t_max'],
}

#starting conditions -> wrap this to a function or something
start_state = np.array([states['u'], 
                        states['w'], 
                        states['q'], 
                        states['theta']])


start_control = np.array([controls['delta_e'], 
                          controls['delta_t']])

#terminal conditions
goal_state = np.array([28, 
                       0, 
                       np.deg2rad(0), 
                       np.deg2rad(0)])

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

# get global position of the aircraft
u_vector = np.array(state_results['u'])
w_vector = np.array(state_results['w'])
theta_vector = np.array(state_results['theta'])

x_original = 0
y_original = 0
z_original = 0

x_ned = []
y_ned = []
z_ned = []
for i in range(len(u_vector)):
    u = u_vector[i]
    w = w_vector[i]
    theta = theta_vector[i]
    
    R = euler_dcm_body_to_inertial(0, theta, 0)
    body_vel = np.array([u, 0, w])
    inertial_vel = np.matmul(R, body_vel)
    
    inertial_pos = inertial_vel * mpc_params['dt_val']
    inertial_pos = inertial_pos + np.array([x_original, y_original, z_original])

    x_ned.append(inertial_pos[0])
    y_ned.append(inertial_pos[1])
    z_ned.append(inertial_pos[2])

    x_original = inertial_pos[0]
    y_original = inertial_pos[1]
    z_original = inertial_pos[2]

time_vec = np.arange(0, mpc_params['dt_val']*(len(u_vector)), 
                        mpc_params['dt_val'])

#drop last element of time_vec
print("len time_vec: ", len(time_vec))

#plot the results
fig,ax = plt.subplots(5,1, figsize=(10,10))
ax[0].plot(time_vec,state_results['u'], label='u')
ax[0].set_ylabel('u (m/s)')

ax[1].plot(time_vec,state_results['w'], label='w')
ax[1].set_ylabel('w (m/s)')

ax[2].plot(time_vec,np.rad2deg(state_results['q']), label='q')
ax[2].set_ylabel('q (rad/s)')

ax[3].plot(time_vec,np.rad2deg(state_results['theta']), label='theta')
ax[3].set_ylabel('theta (deg)')

# ax[4].plot(time_vec,np.rad2deg(state_results['h']), label='h')
# ax[4].set_ylabel('h (deg)')

#plot time_vec,controls
time_vec = time_vec[:-1]

fig,ax = plt.subplots(2,1, figsize=(10,10))
ax[0].plot(time_vec,np.rad2deg(control_results['delta_e']), label='delta_e')
ax[0].set_ylabel('delta_e (deg)')
ax[1].plot(time_vec,control_results['delta_t'], label='delta_t')
ax[1].set_ylabel('delta_t')


#plot time_vec,position in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_ned, y_ned, z_ned)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Position of Aircraft in 3D in NED Frame')
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
    
    

