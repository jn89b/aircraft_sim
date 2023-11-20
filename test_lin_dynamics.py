import numpy as np
import pandas as pd
import casadi as ca
import pickle as pkl
import matplotlib.pyplot as plt

from src.aircraft.AircraftDynamics import LinearizedAircraft
from src.Utils import get_airplane_params
from src.mpc.FixedWingMPC import LateralMPC
from src.math_lib.VectorOperations import euler_dcm_body_to_inertial, euler_dcm_inertial_to_body, \
    compute_B_matrix
    
df = pd.read_csv("SIM_Plane_h_vals.csv")
airplane_params = get_airplane_params(df)
linear_aircraft = LinearizedAircraft(airplane_params)

init_states = {
    'u': 25,
    'w': 0.924,
    'q': 0,
    'theta': np.deg2rad(-0.03),
    'h': 0.0,
    'x': 0.0,
    'v': 0,
    'p': 0,
    'r': 0,
    'phi': np.deg2rad(0.0),
    'psi': 0.0,
    'y': 0.0,
}

input_elevator = np.deg2rad(9.67)
input_throttle = 0.124
input_aileron = np.deg2rad(0)
input_rudder = np.deg2rad(0)
init_controls = {
    'delta_e': input_elevator,
    'delta_t': input_throttle,
    'delta_a': input_aileron,
    'delta_r': input_rudder,
}

A_lon = linear_aircraft.compute_A_lon(init_states['u'],
                                      init_states['theta'],
                                      True,
                                      init_states['w'])

A_lat = linear_aircraft.compute_A_lat(init_states['u'],
                                      init_states['theta'])

A_full = linear_aircraft.compute_A_full(init_states['u'],
                                        init_states['theta'],
                                        init_states['w'])


B_lon = linear_aircraft.compute_B_lon(init_states['u'])
B_lat = linear_aircraft.compute_B_lat(init_states['u'])
B_full = linear_aircraft.compute_B_full(init_states['u'])
eigen_values = np.linalg.eig(A_full)[0]

with open('A_full.pkl', 'wb') as f:
    pkl.dump(A_full, f)
    
with open('B_full.pkl', 'wb') as f:
    pkl.dump(B_full, f)

states = np.array([  init_states['u'],
                     init_states['w'],
                     init_states['q'],
                     init_states['theta'],
                     init_states['h'],
                     init_states['x'],
                     init_states['v'],
                     init_states['p'],
                     init_states['r'],
                     init_states['phi'],
                     init_states['psi'],
                     init_states['y']])

controls = np.array([init_controls['delta_e'],
                        init_controls['delta_t'],
                        init_controls['delta_a'],
                        init_controls['delta_r']])

dt = 0.01
t_final = 30
N = int(t_final / dt)

state_history = []
control_history = []
delta_x_history = []

for i in range(N):
    
    #if odd number, input aileron
    # if i % 10 == 0:
    #     input_aileron = np.deg2rad(-10)
    #     input_rudder = np.deg2rad(0)
    # else:
    # input_aileron = np.deg2rad(10)
    # input_rudder = np.deg2rad(0)
    input_aileron = 0
    input_rudder = 0
    states = linear_aircraft.rk45(input_elevator, input_throttle,
                                  input_aileron, input_rudder,
                                  states, A_full, B_full, dt)
    
    state_history.append(states)
    control_history.append(controls)
    
    #compute the delta x
    delta_x = states - states
    delta_x_history.append(delta_x)
    
state_history = np.array(state_history)
control_history = np.array(control_history)

#plot the results
time_vec = np.linspace(0, t_final, N)
fig,ax = plt.subplots(6,2)
ax[0,0].plot(time_vec, state_history[:,0])
#ax[0,0].("u (m/s)")
ax[0,0].set_ylabel("u (m/s)")
ax[1,0].plot(time_vec, state_history[:,1])
ax[1,0].set_ylabel("w (m/s)")
ax[2,0].plot(time_vec, np.rad2deg(state_history[:,2]))
ax[2,0].set_ylabel("q (m/s)")
ax[3,0].plot(time_vec, np.rad2deg(state_history[:,3]))
ax[3,0].set_ylabel("theta (deg)")
ax[4,0].plot(time_vec, state_history[:,4])
ax[4,0].set_ylabel("h (m)")
ax[5,0].plot(time_vec, state_history[:,5])
ax[5,0].set_ylabel("x (m)")

#have plots share x axis
for i in range(6):
    ax[i,0].sharex(ax[i,1])


ax[0,1].plot(time_vec, state_history[:,6])
ax[0,1].set_ylabel("v (m/s)")
ax[1,1].plot(time_vec, np.rad2deg(state_history[:,7]))
ax[1,1].set_ylabel("p (deg/s)")
ax[2,1].plot(time_vec, np.rad2deg(state_history[:,8]))
ax[2,1].set_ylabel("r (deg/s)")
ax[3,1].plot(time_vec, np.rad2deg(state_history[:,9]))
ax[3,1].set_ylabel("phi (deg)")
ax[4,1].plot(time_vec, np.rad2deg(state_history[:,10]))
ax[4,1].set_ylabel("psi (deg)")
ax[5,1].plot(time_vec, state_history[:,11])
ax[5,1].set_ylabel("y (m)")

#plot control history
fig,ax = plt.subplots(4,1)
ax[0].plot(time_vec, np.rad2deg(control_history[:,0]))
ax[0].set_ylabel("delta_e (deg)")
ax[1].plot(time_vec, control_history[:,1])
ax[1].set_ylabel("delta_t")
ax[2].plot(time_vec, np.rad2deg(control_history[:,2]))
ax[2].set_ylabel("delta_a (deg)")
ax[3].plot(time_vec, np.rad2deg(control_history[:,3]))
ax[3].set_ylabel("delta_r (deg)")



plt.show()


#plot the eigen values
# plt.scatter(np.real(eigen_values), np.imag(eigen_values))
# plt.show()

# #find how many 0's are in the A matrix  
# num_zeros = A_full.size - np.count_nonzero(A_full)

# print("Number of zeros:", num_zeros)
