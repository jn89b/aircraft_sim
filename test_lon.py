import numpy as np
import pandas as pd
import casadi as ca

import matplotlib.pyplot as plt

from src.Utils import read_lon_matrices, get_airplane_params
from src.aircraft.AircraftDynamics import LonAirPlaneCasadi
from src.mpc.FixedWingMPC import LongitudinalMPC

folder_dir = "src/aircraft/derivative_info/"
file_name = "example.txt"

full_path = folder_dir + file_name
A_lon,B_lon = read_lon_matrices(full_path)

df = pd.read_csv("Coeffs.csv")
airplane_params  = get_airplane_params(df)

v_trim = 25
lon_aircraft = LonAirPlaneCasadi(A_lon, B_lon, v_trim)
lon_aircraft.set_state_space()

dt_val = 0.01
Q = np.diag([1, 1, 1, 1, 1])
R = np.diag([1, 1])

mpc_params = {
    'model': lon_aircraft,
    'dt_val': 0.1,
    'N': 10,
    'Q': Q,
    'R': R,
}

lon_mpc_constraints = {
    'delta_e_min': -np.deg2rad(30),
    'delta_e_max': np.deg2rad(30),
    'delta_t_min': 0.1,
    'delta_t_max': 1.0,
    'u_min': 15,
    'u_max': 30,
    'w_min': -10,
    'w_max': 10,
    'q_min': np.deg2rad(-30),
    'q_max': np.deg2rad(30),
    'theta_min': -np.deg2rad(30),
    'theta_max': np.deg2rad(30),
    'h_min': 75,
    'h_max': 150,
}


states = {
    'u': 25,
    'w': 0,
    'q': 0,
    'theta': 0,
    'h': 100,
}

controls = {
    'delta_e': 0,
    'delta_t': 0.5,
}

#starting conditions -> wrap this to a function or something
start_state = np.array([states['u'], states['w'], states['q'], states['theta'], states['h']])
start_control = np.array([controls['delta_e'], controls['delta_t']])

#terminal conditions
goal_state = np.array([25, 0, 0, 0, 100])

#begin mpc
lon_mpc = LongitudinalMPC(mpc_params, lon_mpc_constraints)

lon_mpc.initDecisionVariables()
lon_mpc.reinitStartGoal(start_state, goal_state)
lon_mpc.computeCost()
lon_mpc.defineBoundaryConstraints()
lon_mpc.addAdditionalConstraints()

control_results, state_results = lon_mpc.solveMPCRealTimeStatic(
    start_state, goal_state, start_control)



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
    
    

