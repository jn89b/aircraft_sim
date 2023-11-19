
from src.aircraft.AircraftDynamics import LonAirPlane
from src.Utils import read_lon_matrices, get_airplane_params
from src.config import Config
import pandas as pd
import numpy as np
import control
import control.matlab

df = pd.read_csv("SIM_Plane_h_vals.csv")
airplane_params = get_airplane_params(df)

lon_airplane = LonAirPlane(airplane_params)

init_states = {
    'u': 25,
    'w': 0.729,
    'q': 0,
    'theta': np.deg2rad(8),
    'h': 0.0,
}

delta_e_cmd = np.deg2rad(10.0)
init_controls = {
    'delta_e': np.deg2rad(delta_e_cmd),
    'delta_t': 0.128,
}

A = lon_airplane.compute_A(init_states['u'], 
                           init_states['theta'],
                           True,
                           init_states['w'])

B = lon_airplane.compute_B(init_states['u'])

states = np.array([init_states['u'],
                     init_states['w'],
                     init_states['q'],
                     init_states['theta'],
                     init_states['h']])

print("A",A)
print("B",B)

controls = np.array([init_controls['delta_e'],
                        init_controls['delta_t']])

#turn into state space model
lon_ss = control.ss(A, B, np.eye(A.shape[0]), np.zeros(B.shape))

timevec = np.linspace(0, 50, 1000)
de = delta_e_cmd * np.ones(timevec.shape)

#map throttle to mass of aircraft
# thrust_scale = airplane_params['thrust_scale']
thrust_scale = airplane_params['mass'] * Config.G / \
    Config.HOVER_THROTTLE * init_controls['delta_t']
print("thrust_scale:", thrust_scale)
print("weight of aircraft:", airplane_params['mass'] * Config.G)
dthrottle = np.ones(timevec.shape) * thrust_scale
inputs = np.vstack((de, dthrottle))

Time, [u,w,q,theta,h] = control.forced_response(
    lon_ss, U=inputs, T=timevec, X0=states)

q = np.rad2deg(q)
theta = np.rad2deg(theta)

#plotting
import matplotlib.pyplot as plt
fig,ax = plt.subplots(2,2, figsize=(10,10))

ax[0,0].plot(Time, u)
ax[0,0].set_xlabel("Time (s)")
ax[0,0].set_ylabel("u (m/s)")

ax[0,1].plot(Time, w)
ax[0,1].set_xlabel("Time (s)")
ax[0,1].set_ylabel("w (m/s)")

ax[1,0].plot(Time, q)
ax[1,0].set_xlabel("Time (s)")
ax[1,0].set_ylabel("q (deg/s)")

ax[1,1].plot(Time, theta)
ax[1,1].set_xlabel("Time (s)")
ax[1,1].set_ylabel("theta (deg)")

#plot title with control inputs
fig.suptitle("Control Inputs: delta_e = {} deg, delta_t = {} N".format(
    np.rad2deg(delta_e_cmd), thrust_scale))

#plot height vs time
fig,ax = plt.subplots(1,1, figsize=(10,10))
ax.plot(Time, h)
ax.set_xlabel("Time (s)")
ax.set_ylabel("h (m)")

# #plot the eigenvalues in one plot 
eigenvalues = np.linalg.eigvals(A)
fig,ax = plt.subplots(1,1, figsize=(10,10))
ax.scatter(eigenvalues.real, eigenvalues.imag, marker='x')


plt.show()
"""
- POLE PLACEMENT
- TRANSFER FUNCTION
- SMALL PERTURBATION LINEARIZATION'
"""