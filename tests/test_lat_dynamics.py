from src.aircraft.AircraftDynamics import LatAirPlane
from src.Utils import  get_airplane_params

import pandas as pd
import numpy as np
import pickle as pkl

df = pd.read_csv("SIM_Plane_h_vals.csv")
airplane_params = get_airplane_params(df)
lat_airplane = LatAirPlane(airplane_params)

init_states = {
    'v': 0,
    'p': 0,
    'r': 0,
    'phi': np.deg2rad(0.0),
    'psi': 0.0,
    'y': 0.0,
}

init_controls = {
    'delta_a': np.deg2rad(0.0),
    'delta_r': np.deg2rad(0.0),
}

velocity = 25.0
theta_rad = np.deg2rad(-0.03)

A = lat_airplane.compute_A(velocity, theta_rad)
B = lat_airplane.compute_B(velocity)
#pickle the matrices
with open('A_lat.pkl', 'wb') as f:
    pkl.dump(A, f)
with open('B_lat.pkl', 'wb') as f:
    pkl.dump(B, f)
    
states = np.array([init_states['v'],
                     init_states['p'],
                     init_states['r'],
                     init_states['phi'],
                     init_states['psi'],
                     init_states['y']])

controls = np.array([init_controls['delta_a'],
                        init_controls['delta_r']])

dt = 0.01

t_final = 10
N = int(t_final / dt)

state_history = []
control_history = []

#print("A matrix:", A)
#print("B matrix:", B)

print("A SHAPE:", A.shape)
for i in range(N):
    
    #if odd number, input aileron
    # if i % 10 == 0:
    #     input_aileron = np.deg2rad(-10)
    #     input_rudder = np.deg2rad(0)
    # else:
    # input_aileron = np.deg2rad(10)
    # input_rudder = np.deg2rad(0)
    input_aileron = np.deg2rad(15)
    input_rudder = 0.0
    states = lat_airplane.rk45(input_aileron, input_rudder,
                               states, A, B, dt)
    
    state_history.append(states)
    control_history.append([input_aileron, input_rudder])
    
state_history = np.array(state_history)
control_history = np.array(control_history)

#plot the results
import matplotlib.pyplot as plt

time_vec = np.linspace(0, t_final, N)
fig,ax = plt.subplots(6,1, figsize=(10,10))
ax[0].plot(time_vec, state_history[:,0])
ax[0].set_xlabel('time (s)')
ax[0].set_ylabel('v (m/s)')
ax[0].grid(True)

ax[1].plot(time_vec, state_history[:,1])
ax[1].set_xlabel('time (s)')
ax[1].set_ylabel('p (rad/s)')
ax[1].grid(True)

ax[2].plot(time_vec, np.deg2rad(state_history[:,2]))
ax[2].set_xlabel('time (s)')
ax[2].set_ylabel('r (deg/s)')
ax[2].grid(True)

ax[3].plot(time_vec, np.rad2deg(state_history[:,3]))
ax[3].set_xlabel('time (s)')
ax[3].set_ylabel('phi (deg)')
ax[3].grid(True)

ax[4].plot(time_vec, np.rad2deg(state_history[:,4]))
ax[4].set_xlabel('time (s)')
ax[4].set_ylabel('psi (deg)')
ax[4].grid(True)

ax[5].plot(time_vec, state_history[:,5])
ax[5].set_xlabel('time (s)')
ax[5].set_ylabel('y (m)')
ax[5].grid(True)

#get eigenvalues
eigenvalues = np.linalg.eigvals(A)

#plot the eigenvalues
fig,ax = plt.subplots(1,1, figsize=(10,10))
ax.scatter(np.real(eigenvalues), np.imag(eigenvalues))
ax.set_xlabel('real')
ax.set_ylabel('imaginary')

plt.tight_layout()
plt.show()
    

