
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
    'u': 15,
    'w': 0.729,
    'q': 0,
    'theta': np.deg2rad(8),
    'h': 0.0,
    'x': 0.0,
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
                     init_states['h'],
                     init_states['x']])

print("A",A)
print("B",B)

controls = np.array([init_controls['delta_e'],
                        init_controls['delta_t']])

#turn into state space model
lon_ss = control.ss(A, B, np.eye(A.shape[0]), np.zeros(B.shape))


#design lqr controller 
Q = np.diag([1,   
             0, 
             0, 
             1, 
             0, 
             0])

R = np.diag([0.1, 
             0.1])

K, S, E = control.lqr(lon_ss, Q, R)

u_desired = 25
theta_desired = np.deg2rad(8)
# Xd = np.matrix([[u_desired,0,0,0,0,0],
#              [0,1,0,0,0,0]]).T

#set Xd to be the desired state
Xd = np.matrix([[u_desired,0,0,theta_desired,0,0],
             [0,0,0,theta_desired,0,0]]).T


# Closed loop dynamics
C = np.eye(A.shape[0])
D = np.zeros((A.shape[0], B.shape[1]))
H = control.ss(A-B*K,B*K*Xd,C,D)

# step response
t = np.linspace(0, 10, 1000)
t, y = control.step_response(H, t, X0=states)
#t, x = control.initial_response(H, t, X0=states)


# #plotting
import matplotlib.pyplot as plt
plt.close('all')
#plot system response
plt.figure()
fig, ax = plt.subplots(3,2,figsize=(10,10))
plt.suptitle('LQR Control Response')
ax[0,0].plot(t,x)
ax[0,0].set_xlabel('Time [s]')
ax[0,0].set_ylabel('Velocity [m/s]')
ax[0,0].grid()

ax[0,1].plot(t,y)
ax[0,1].set_xlabel('Time [s]')
ax[0,1].set_ylabel('Velocity [m/s]')
ax[0,1].grid()

# ax[1,0].plot(t,x[:,1])
# ax[1,0].set_xlabel('Time [s]')
# ax[1,0].set_ylabel('Angle [rad]')
# ax[1,0].grid()



plt.show()



"""
- POLE PLACEMENT
- TRANSFER FUNCTION
- SMALL PERTURBATION LINEARIZATION'
"""