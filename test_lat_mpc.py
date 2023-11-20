import numpy as np
import pandas as pd
import casadi as ca
import pickle as pkl
import matplotlib.pyplot as plt

from src.aircraft.AircraftDynamics import LatAirPlane, LatAirPlaneCasadi
from src.Utils import get_airplane_params
from src.mpc.FixedWingMPC import LateralMPC
from src.math_lib.VectorOperations import euler_dcm_body_to_inertial, euler_dcm_inertial_to_body, \
    compute_B_matrix
    
# A_lon,B_lon = read_lon_matrices(full_path)
df = pd.read_csv("Coeffs.csv")
airplane_params = get_airplane_params(df)


#load the matrices
with open('A_lat.pkl', 'rb') as f:
    A_lat = pkl.load(f)
    
with open('B_lat.pkl', 'rb') as f:
    B_lat = pkl.load(f)
    
print("A" , A_lat)
print("B" , B_lat)
lat_aircraft_ca = LatAirPlaneCasadi(airplane_params, True,
                                    A_lat, True, B_lat)

lat_aircraft_ca.set_state_space()


Q = np.diag([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]) 
R = np.diag([0.1, 0.1])

mpc_params = {
    'model': lat_aircraft_ca,
    'dt_val': 0.05,
    'N': 15,
    'Q': Q,
    'R': R,
}

lat_mpc_constraints = {
    'delta_a_min': np.deg2rad(-25),
    'delta_a_max': np.deg2rad(25),
    'delta_r_min': np.deg2rad(-30),
    'delta_r_max': np.deg2rad(30),
    'v_min': -35, #don't need this really
    'v_max': 35,  #don't need this really
    'p_min': np.deg2rad(-60),
    'p_max': np.deg2rad(60),
    'r_min': np.deg2rad(-60),
    'r_max': np.deg2rad(60),
    'phi_min': np.deg2rad(-45),
    'phi_max': np.deg2rad(45),
    # 'psi_min': np.deg2rad(-180),
    # 'psi_max': np.deg2rad(180),
}

states = {
    'v': 0.0,
    'p': 0.0,
    'r': 0.0,
    'phi': 0.0,
    'psi': 0.0,
    'y': 0.0,
}

controls = {
    'delta_a': 0.0,
    'delta_r': 0.0,
}

start_state = np.array([states['v'], 
                        states['p'], 
                        states['r'], 
                        states['phi'], 
                        states['psi'], 
                        states['y']])

start_control = np.array([controls['delta_a'],
                            controls['delta_r']])

#terminal conditions
goal_v = 0.0
goal_p = 0.0
goal_r = 0.0
goal_phi = 0.0
goal_psi = 25.0
goal_y = 0.0
goal_state = np.array([goal_v,
                        goal_p,
                        goal_r,
                        np.deg2rad(goal_phi),
                        np.deg2rad(goal_psi),
                        goal_y])

#begin mpc
lat_mpc = LateralMPC(mpc_params, lat_mpc_constraints)
lat_mpc.initDecisionVariables()
lat_mpc.reinitStartGoal(start_state, goal_state)
lat_mpc.computeCost()
lat_mpc.defineBoundaryConstraints()
lat_mpc.addAdditionalConstraints()

control_results, state_results = lat_mpc.solveMPCRealTimeStatic(
    start_state, goal_state, start_control)

#unpack the results
control_results = lat_mpc.unpack_controls(control_results)
state_results = lat_mpc.unpack_states(state_results)

## simulate the trajectory of the aircraft
t_final = 10 #seconds
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
    
    # if time_current > t_final/2:
    #     new_v = 0
    #     new_p = 0
    #     new_r = 0
    #     new_phi = np.deg2rad(0)
    #     new_psi = np.deg2rad(0)
    #     new_y = 0
        
    #     goal_state = np.array([new_v,
    #                             new_p,
    #                             new_r,
    #                             new_phi,
    #                             new_psi,
    #                             new_y])
        
    lat_mpc.reinitStartGoal(start_state, goal_state)
    
    control_results, state_results = lat_mpc.solveMPCRealTimeStatic(
        start_state, goal_state, start_control)
    
    #unpack the results
    control_results = lat_mpc.unpack_controls(control_results)
    state_results = lat_mpc.unpack_states(state_results)
    
    #update the start state
    start_state = np.array([state_results['v'][idx_start],
                            state_results['p'][idx_start],
                            state_results['r'][idx_start],
                            state_results['phi'][idx_start],
                            state_results['psi'][idx_start],
                            state_results['y'][idx_start]]
                           )
    start_control = np.array([control_results['delta_a'][idx_start],
                                control_results['delta_r'][idx_start]])
    
    #store the results
    control_history.append(start_control)
    state_history.append(start_state)
    
    #update the time
    time_current += mpc_params['dt_val']
    

v = [x[0] for x in state_history]
p = [x[1] for x in state_history]
r = [x[2] for x in state_history]
phi = [x[3] for x in state_history]
psi = [x[4] for x in state_history]
y = [x[5] for x in state_history]

delta_a = [x[0] for x in control_history]
delta_r = [x[1] for x in control_history]

time_vec = np.arange(0, t_final, mpc_params['dt_val'])

#%% 
fig,ax = plt.subplots(6,1, figsize=(10,10))
ax[0].plot(time_vec, v, label='v')
ax[1].plot(time_vec, np.rad2deg(p), label='p')
ax[2].plot(time_vec, np.rad2deg(r), label='r')
ax[3].plot(time_vec, np.rad2deg(phi), label='phi')
ax[4].plot(time_vec, np.rad2deg(psi), label='psi')
ax[5].plot(time_vec, y, label='y')

ax[0].set_ylabel('v')
ax[1].set_ylabel('p')
ax[2].set_ylabel('r')
ax[3].set_ylabel('phi')
ax[4].set_ylabel('psi')
ax[5].set_ylabel('y')

ax[0].legend()
ax[1].legend()
ax[2].legend()
ax[3].legend()
ax[4].legend()
ax[5].legend()


fig,ax = plt.subplots(2,1, figsize=(10,10))
ax[0].plot(time_vec, np.rad2deg(delta_a), label='delta_a')
ax[1].plot(time_vec, np.rad2deg(delta_r), label='delta_r')

ax[0].set_ylabel('delta_a')
ax[1].set_ylabel('delta_r')

ax[0].legend()
ax[1].legend()

plt.show()




