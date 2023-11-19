from src.aircraft.AircraftDynamics import LonAirPlane
from src.Utils import read_lon_matrices, get_airplane_params
import pandas as pd
import numpy as np
import pickle as pkl
#https://aircraftflightmechanics.com/Linearisation/LinearvsNonlinear.html

df = pd.read_csv("SIM_Plane_h_vals.csv")
airplane_params = get_airplane_params(df)
lon_airplane = LonAirPlane(airplane_params)

init_states = {
    'u': 25,
    'w': 0.924,
    'q': 0,
    'theta': np.deg2rad(-0.002),
    'h': 0.0,
}

delta_e_cmd = 9.67
max_input = delta_e_cmd

init_controls = {
    'delta_e': np.deg2rad(delta_e_cmd),
    'delta_t': 0.124,
}

A = lon_airplane.compute_A(init_states['u'], 
                           init_states['theta'],
                           True,
                           init_states['w'])

B = lon_airplane.compute_B(init_states['u'])

#pickle the matrices
with open('A.pkl', 'wb') as f:
    pkl.dump(A, f)
    
with open('B.pkl', 'wb') as f:
    pkl.dump(B, f)

states = np.array([init_states['u'],
                     init_states['w'],
                     init_states['q'],
                     init_states['theta'],
                     init_states['h']])

controls = np.array([init_controls['delta_e'],
                        init_controls['delta_t']])

dt = 0.01
t_final = 30
N = int(t_final / dt)

state_history = []
control_history = []
delta_x_history = []

print("A matrix:", A)
print("B matrix:", B)

Ax = np.matmul(A, states)
print("Ax:", Ax)
Bu = np.matmul(B, controls)
print("Bu:", Bu)
eigen_history = []

current_time = 0

for i in range(N):
    
    #at first third of N set elevator to 
    if current_time <= t_final/3:
        delta_e_cmd = np.deg2rad(max_input)
    #at second third of N set elevator to max_input
    elif (current_time <= 2*t_final/3) and (current_time > t_final/3):
        delta_e_cmd = np.deg2rad(max_input)
    #at last third of N set elevator to 0
    else:
        delta_e_cmd = np.deg2rad(max_input)
    
    
    # A = lon_airplane.compute_A(states[0],
    #                              states[3],
    #                              True,
    #                              states[1])
    
    # B = lon_airplane.compute_B(states[0])

    controls = np.array([delta_e_cmd,
                        init_controls['delta_t']])
        
    states = lon_airplane.rk45(delta_e_cmd, init_controls['delta_t'], states, 
                               A, B, dt)
    
    state_history.append(states)
    control_history.append(controls)
    
    delta_x = np.matmul(A, states) + np.matmul(B, controls)
    
    # B = lon_airplane.compute_B(states[0])
    
    eigenvalues = np.linalg.eigvals(A)
    
    eigen_history.append(eigenvalues)
    delta_x_history.append(delta_x)
    
    
    current_time += dt
    
# plot the results
import matplotlib.pyplot as plt
plt.close('all')
state_history = np.array(state_history)
control_history = np.array(control_history)
delta_x_history = np.array(delta_x_history)

fig,axis = plt.subplots(5,1, figsize=(10,10))
axis[0].plot(state_history[:,0], label='u')
axis[1].plot(state_history[:,1], label='w')
axis[2].plot(np.rad2deg(state_history[:,2]), label='q')
axis[3].plot(np.rad2deg(state_history[:,3]), label='theta')
axis[4].plot(state_history[:,4], label='h')
axis[0].legend()
axis[1].legend()
axis[2].legend()
axis[3].legend()
axis[4].legend()

#plot delta_x history
fig,ax = plt.subplots(5,1, figsize=(10,10))
ax[0].plot(delta_x_history[:,0], label='delta_x')
ax[0].set_ylabel('delta_x (m/s)')
ax[1].plot(delta_x_history[:,1], label='delta_w')
ax[1].set_ylabel('delta_w (m/s)')
ax[2].plot(np.rad2deg(delta_x_history[:,2]), label='delta_q')
ax[2].set_ylabel('delta_q (deg/s)')
ax[3].plot(np.rad2deg(delta_x_history[:,3]), label='delta_theta')
ax[3].set_ylabel('delta_theta (deg)')
ax[4].plot(delta_x_history[:,4], label='delta_h')
ax[4].set_ylabel('delta_h (m)')

#plot time_vec,controls
time_vec = np.arange(0, len(control_history)*dt, dt)
fig,ax = plt.subplots(2,1, figsize=(10,10))
ax[0].plot(time_vec,np.rad2deg(control_history[:,0]), label='delta_e')
ax[0].set_ylabel('delta_e (deg)')
ax[1].plot(time_vec,control_history[:,1], label='delta_t')
ax[1].set_ylabel('delta_t')



#plot the eigenvalues in one plot as a function of time
eigen_history = np.array(eigen_history)
plt.figure()
plt.plot(eigen_history.real, eigen_history.imag, 'x')
plt.xlabel('real')
plt.ylabel('imag')

plt.show()
