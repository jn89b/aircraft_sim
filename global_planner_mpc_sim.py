"""
Example of a global planner for the MPC simulation
"""
import numpy as np
import pandas as pd
import casadi as ca
import pickle as pkl
import matplotlib.pyplot as plt
import math as m

from src.Utils import read_lon_matrices, get_airplane_params
from src.aircraft.AircraftDynamics import LinearizedAircraft, LinearizedAircraftCasadi
from src.mpc.FixedWingMPC import LinearizedAircraftMPC
from src.math_lib.VectorOperations import euler_dcm_body_to_inertial, euler_dcm_inertial_to_body, \
    compute_B_matrix
    
from src.aircraft.AircraftDynamics import add_noise_to_linear_states


""" 

First off we need to load the airplane parameter coefficients as well 
as the linearized matrices 
"""
df = pd.read_csv("SIM_Plane_h_vals.csv")
airplane_params = get_airplane_params(df)
with open('A_full.pkl', 'rb') as f:
    A_full = pkl.load(f)
    
with open('B_full.pkl', 'rb') as f:
    B_full = pkl.load(f)
    
lin_aircraft_ca = LinearizedAircraftCasadi(airplane_params, 
                                           A_full, 
                                           B_full)


"""
Next we need to set up the MPC, this is a simple example of how to do it
"""
#weighting matrices for state the higher the value the more the 
# state is penalized (we care more about it the higher it is) 
Q = np.diag(
[   1.0, #u
    0.0, #w
    0.0, #q
    0.0, #theta
    1.0, #h
    0.0, #x
    0.0, #v
    0.0, #p
    0.0, #r
    0.0, #phi
    1.0, #psi
    0.0, #y
])

# weighting matrices for the control the higher the value the more the
# the higher the value the more we penalize the control
R = np.diag([
    0.1, #delta_e
    0.1, #delta_t
    0.1, #delta_a
    0.1, #delta_r
])

mpc_params = {
    'model': lin_aircraft_ca, # the linearized aircraft model
    'dt_val': 0.05, #time step for the MPC
    'N': 15, #prediction horizon
    'Q': Q, #state weighting matrix
    'R': R, #control weighting matrix
}

# this are the constrints we give to our controller 
lin_mpc_constraints = {
    'delta_e_min': np.deg2rad(-30),
    'delta_e_max': np.deg2rad(30),
    'delta_t_min': 0.1,
    'delta_t_max': 0.75,
    'delta_a_min': np.deg2rad(-25),
    'delta_a_max': np.deg2rad(25),
    'delta_r_min': np.deg2rad(-30),
    'delta_r_max': np.deg2rad(30),
    'u_min': 15,
    'u_max': 35,
    'w_min': -0.5,
    'w_max': 0.5,
    'q_min': np.deg2rad(-60),
    'q_max': np.deg2rad(60),
    'theta_min': np.deg2rad(-35),
    'theta_max': np.deg2rad(35),
    'v_min': -35, #don't need this really
    'v_max': 35,  #don't need this really
    'p_min': np.deg2rad(-60),
    'p_max': np.deg2rad(60),
    'r_min': np.deg2rad(-60),
    'r_max': np.deg2rad(60),
    'phi_min': np.deg2rad(-60),
    'phi_max': np.deg2rad(60)
}

# this is the initial state of the aircraft
states = {
    'u': 25,
    'w': 0.0,
    'q': 0,
    'theta': np.deg2rad(-0.03),
    'h': -3.0,
    'x': 0.0,
    'v': 0.0,
    'p': 0.0,
    'r': 0.0,
    'phi': 0.0,
    'psi': np.deg2rad(45.0),
    'y': 0.0,
}

start_state = np.array([states['u'],
                        states['w'],
                        states['q'],
                        states['theta'],
                        states['h'],
                        states['x'],
                        states['v'],
                        states['p'],
                        states['r'],
                        states['phi'],
                        states['psi'],
                        states['y']])

# this is the initial control input
controls = {
    'delta_e': np.deg2rad(0), #elevator in radians
    'delta_t': 0.1, #throttle in percent
    'delta_a': np.deg2rad(0), #aileron in radians
    'delta_r': np.deg2rad(0), #rudder in radians
}

start_control = np.array([controls['delta_e'],
                            controls['delta_t'],
                            controls['delta_a'],
                            controls['delta_r']])


goal_x = 50
goal_y = 60
goal_z = 70  
goal_u = 25
goal_v = 0
goal_w = 0
goal_theta = np.deg2rad(0)
goal_phi = np.deg2rad(0)
goal_psi = np.deg2rad(0)
goal_p = 0
goal_q = 0
goal_r = 0

goal_state = np.array([goal_u,
                          goal_w,
                          goal_q,
                          goal_theta,
                          goal_z,
                          goal_x,
                          goal_v,
                          goal_p,
                          goal_r,
                          goal_phi,
                          goal_psi,
                          goal_y])

# now lets create the MPC controller
# you don't need to call the following again once you have created the controller
lin_mpc = LinearizedAircraftMPC(mpc_params, lin_mpc_constraints)
lin_mpc.initDecisionVariables()
lin_mpc.reinitStartGoal(start_state, goal_state)
lin_mpc.computeCost()
lin_mpc.defineBoundaryConstraints()
lin_mpc.addAdditionalConstraints()

# once you are done with the MPC controller you can call the solve function
control_results, state_results = lin_mpc.solveMPCRealTimeStatic(
    start_state, goal_state, start_control)
#use the unpack method to get the states and controls
control_results = lin_mpc.unpack_controls(control_results)
state_results = lin_mpc.unpack_states(state_results)

# Now let's call out the global planner 
