import pandas as pd
import numpy as np
import control
import control.matlab
import matplotlib.pyplot as plt

from scipy.signal import ss2tf

"""
https://ctms.engin.umich.edu/CTMS/index.php?example=AircraftPitch&section=SystemModeling

If you are measuring everything in your system, it means that all state variables are directly observable. In this case, the CC matrix would be an identity matrix, where each state variable contributes directly to the corresponding output. The size of CC would be m×nm×n, where mm is the number of outputs, and nn is the number of state variables.

Mathematically, the CC matrix would look like this
"""

## test example 
A = np.array([[-0.313, 56.7, 0.0],
              [-0.0139, -0.426, 0.0],
              [0 , 56.7, 0]])

#column vector for elevator
B = np.array([[0.232],
              [0.0203],
              [0]])
num,den = ss2tf(A, B, np.eye(A.shape[0]), np.zeros(B.shape))

pitch_num = num[2]

#get the transfer function of pitch angle to elevator
tf = control.tf(pitch_num, den)
print(tf)

#let's design an lqr controller
#first do a 5 degree pitch step
initial_conditions = np.array([0, 0, 0])
step_cmd = np.deg2rad(25)
time_vec = np.linspace(0, 35, 1000)
yout,t= control.matlab.step(step_cmd*tf, time_vec)

#plot the response
plt.close('all')
fig,ax = plt.subplots(1,1, figsize=(10,10))

ax.plot(time_vec, yout)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Pitch Angle (deg)')

#check if the system is controllable
cntrl = control.ctrb(A,B)
rank = np.linalg.matrix_rank(cntrl)

if rank == A.shape[0]:
    print("System is controllable")
    
#design lqr controller
p = 1
Q = np.dot(cntrl.T, cntrl) * p
# Q = np.diag([0,  #angle of attack 
#              0,  #pitch rate
#              2.0]) #pitch angle
print("Q",Q)
R = np.diag([1])

K, S, E = control.lqr(A, B, Q, R)
print("K",K)
#simulate the closed loop system
C = np.diag([0,0,1])
sys_cls = control.ss(A-B*K, B, C, np.zeros(B.shape))
#do a step response
yout,t = control.matlab.step(step_cmd*sys_cls, time_vec)
#get the response of the pitch angle
alpha = yout[:,0]
pitch_rate = yout[:,1]
theta = yout[:,2]

#plot the response
fig,ax = plt.subplots(1,1, figsize=(10,10))
ax.plot(time_vec, np.rad2deg(alpha), label='alpha')
ax.plot(time_vec, np.rad2deg(pitch_rate), label='pitch rate')
ax.plot(time_vec, np.rad2deg(theta), label='theta')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Pitch Angle (deg)')
ax.legend()
plt.show()