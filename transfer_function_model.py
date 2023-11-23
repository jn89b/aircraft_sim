"""
http://www.ece.ualberta.ca/~tchen/ctm/examples/pitch/Mpitch.html
"""

from src.aircraft.AircraftDynamics import LonAirPlane
from src.Utils import read_lon_matrices, get_airplane_params
from src.config import Config
import pandas as pd
import numpy as np
import control
import control.matlab
import matplotlib.pyplot as plt
from scipy.signal import ss2tf

df = pd.read_csv("SIM_Plane_h_vals.csv")
airplane_params = get_airplane_params(df)
lon_airplane = LonAirPlane(airplane_params)

init_states = {
    'u': 20,
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
C = np.eye(A.shape[0])
#measure everything
num,den = ss2tf(A, B, C, np.zeros(B.shape))

#print out the transfer function as polynomial
pitch_num = num[3]
elevator_den = den

pitch_to_elevator_tf = control.tf(pitch_num, elevator_den)
print("pitch to elevator ", pitch_to_elevator_tf)

#look at the open loop response
pitch_cmd = np.deg2rad(25.0)
time_vec = np.linspace(0, 10, 1000)
yout, t = control.matlab.step(pitch_cmd*pitch_to_elevator_tf, time_vec)

#plot response 
plt.close('all')
fig,ax = plt.subplots(1,1, figsize=(10,10))
ax.plot(time_vec, yout)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Pitch Angle (deg)')


#check if the system is controllable
cntrl = control.ctrb(A,B)
rank = np.linalg.matrix_rank(cntrl)

if rank == A.shape[0]:
    print("System is controllable", rank)
else:
    print("System is not controllable", rank)
    
#design lqr controller
C = np.eye(A.shape[0])
p = 50
Q = np.dot(C.T, C) * p
Q = np.diag([1.0,  #u
             0,  #w
             0.0, #q
             1.0,  #theta
             0,  #h
            0]) #x
R = np.diag([1.0, 10.0])

K,S,E = control.lqr(A, B, Q, R)
print("K",K)
#simulate the closed loop system
C = np.diag([1,1,1,1,1,1])
sys_cl = control.ss(A-B*K, B, C, np.zeros(B.shape))
yout,t = control.matlab.step(pitch_cmd*sys_cl, time_vec)

#simulate a for loop to get the response of the system
dt = 0.01
t_final = 10
N = int(t_final / dt)
state_history = []
control_history = []

goal_state = np.array([25, 0, 0, pitch_cmd, 0, 0])
B = np.array(B)
K = np.array(K)
for i in range(N):
    #print("states are", states)
    #use the gains to compute the control
    error = goal_state - states
    controls = np.matmul(K, error)
    #multiply the 
    controls = controls.T
    #scale thrust to weight
    #constrain the throttle to not be negative
    if controls[1] <= 0.1:
        print("throttle is too low")
        controls[1] = 0.1

    controls[1] = controls[1] * airplane_params['mass'] * Config.G \
                    / Config.HOVER_THROTTLE
                
    x_dot = np.matmul(A,states) + np.matmul(B, controls)
    states = states + (x_dot * dt)

    state_history.append(states)
    control_history.append(controls)


#plot the simulation
state_history = np.array(state_history)
control_history = np.array(control_history)

fig,axis = plt.subplots(6,1, figsize=(10,10))
axis[0].plot(state_history[:,0], label='u')
axis[1].plot(state_history[:,1], label='w')
axis[2].plot(state_history[:,2], label='q')
axis[3].plot(np.rad2deg(state_history[:,3]), label='theta')
axis[3].plot(np.rad2deg(pitch_cmd*np.ones(len(time_vec))),
                label='theta_cmd', linestyle='--')
axis[4].plot(state_history[:,4], label='h')
axis[5].plot(state_history[:,5], label='x')

for ax in axis:
    ax.legend()

#plot controls
fig,axis = plt.subplots(2,1, figsize=(10,10))
axis[0].plot(control_history[:,0], label='delta_e')
axis[1].plot(control_history[:,1], label='delta_t')

for ax in axis:
    ax.legend()
    

# #get response of system
# u = yout[:,0]
# w = yout[:,1]
# q = yout[:,2]
# theta = yout[:,3]
# h = yout[:,4]
# x = yout[:,5]

# #plot the response
# fig,ax = plt.subplots(6,1, figsize=(10,10))
# ax[0].plot(time_vec, u, label='u')
# ax[1].plot(time_vec, w, label='w')
# ax[2].plot(time_vec, np.rad2deg(q), label='q')
# ax[3].plot(time_vec, np.rad2deg(theta), label='theta')
# ax[3].plot(time_vec, -np.rad2deg(pitch_cmd*np.ones(len(time_vec))), 
#            label='theta_cmd', linestyle='--')
# ax[4].plot(time_vec, h, label='h')
# ax[5].plot(time_vec, x, label='x')

# ax[0].legend()
# ax[1].legend()
# ax[2].legend()
# ax[3].legend()
# ax[4].legend()
# ax[5].legend()

plt.show()


