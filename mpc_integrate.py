"""
Example of a global planner for the MPC simulation
"""
import numpy as np
import pandas as pd
import casadi as ca
import pickle as pkl
import matplotlib.pyplot as plt
import math as m

### Global Planner ###
from src.guidance_lib.src.PositionVector import PositionVector
from src.guidance_lib.src.Grid import Grid, FWAgent
from src.guidance_lib.src.Radar import Radar
from src.guidance_lib.src.Obstacle import Obstacle
from src.guidance_lib.src.SparseAstar import SparseAstar

### Guidance MPC ###
from src.Utils import read_lon_matrices, get_airplane_params
from src.aircraft.AircraftDynamics import LinearizedAircraft, LinearizedAircraftCasadi
from src.mpc.FixedWingMPC import LinearizedAircraftMPC
from src.math_lib.VectorOperations import euler_dcm_body_to_inertial, euler_dcm_inertial_to_body, \
    compute_B_matrix    
from src.aircraft.AircraftDynamics import add_noise_to_linear_states

class State:
    location: list[float]  # x, y, z where y is up, in meters
    rotation: list[float]  # roll, pitch, yaw in radians
    velocity: list[float]  # x, y, z where y is up, in meters per second

    def __init__(self, location: list[float], rotation: list[float], velocity: list[float]):
        self.location = location
        self.rotation = rotation
        self.velocity = velocity

def load_model() -> LinearizedAircraftCasadi:
    df = pd.read_csv("SIM_Plane_h_vals.csv")
    airplane_params = get_airplane_params(df)
    with open('A_full.pkl', 'rb') as f:
        A_full = pkl.load(f)
        
    with open('B_full.pkl', 'rb') as f:
        B_full = pkl.load(f)
        
    lin_aircraft_ca = LinearizedAircraftCasadi(airplane_params, 
                                            A_full, 
                                            B_full)

    return lin_aircraft_ca

def load_mpc(lin_aircraft_ca:LinearizedAircraft,
             start_state:np.ndarray, 
             end_state:np.ndarray) -> LinearizedAircraftMPC:
    #weighting matrices for state the higher the value the more the 
    # state is penalized (we care more about it the higher it is) 
    Q = np.diag([
        0.75, #u
        0.0, #w
        0.0, #q
        1.0, #theta
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
    r_control = 0.4
    R = np.diag([
        r_control, #delta_e
        1.0, #delta_t
        r_control, #delta_a
        r_control, #delta_r
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
    
    # now lets create the MPC controller
    # you don't need to call the following again once you have created the controller
    lin_mpc = LinearizedAircraftMPC(mpc_params, lin_mpc_constraints)
    lin_mpc.initDecisionVariables()
    lin_mpc.reinitStartGoal(start_state, end_state)
    lin_mpc.computeCost()
    lin_mpc.defineBoundaryConstraints()
    lin_mpc.addAdditionalConstraints()
    
    return lin_mpc
    
def get_waypoint_states(path: list[list[float]], 
                        max_speed:float) -> list[State]:
    """
    Helper function to convert the path to a list of states
    """
    waypoint_states = []
    for i in range(len(path)):
        psi = np.deg2rad(path[i][5])
        z = path[i][2]
        pitch_rad = np.deg2rad(path[i][3])
        roll_rad = np.deg2rad(path[i][4])
        position = [path[i][0], path[i][1], z]
        attitude = [roll_rad, pitch_rad, psi]
        #I don't have time to take the unit vector for the airspeed and break it into i, j, k
        inertial_vel = [max_speed, 0, 0]
        waypoint_states.append(State(position, attitude, inertial_vel))

    return waypoint_states
    
def load(
    location: list[float],
    rotation: list[float],
    velocity: list[float],
    target_location: list[float],
    max_speed: float,
    max_yaw_turn_dg: float = 25,
    max_climb_angle_dg: float = 10,
    leg_m:float = 100,
    x_bounds: list[float] = [0, 500], 
    y_bounds: list[float] = [0, 500],
    z_bounds: list[float] = [0, 500],
) -> list[State]:
    # location: Vector3 = Vector3(location)
    # rotation: Rotation = Rotation(rotation)
    # velocity: Vector3 = Vector3(velocity)
    # target_location: Vector3 = Vector3(target_location)
    # Justin -max_speed: float = max_speed # meters per second
    # Justin -max_yaw_turn_dg: float the maximum angle the agent can turn in degrees
    # Justin -max_climb_angle_dg: float = max_climb_angle_dg, the maximum angle the agent can climb in degrees
    # Justin -leg_m: float = leg_m, the distance between waypoints in meters
    # Justin -x_bounds: list[float] = x_bounds, the bounds of the x axis in meters for the grid
    # Justin -y_bounds: list[float] = y_bounds, the bounds of the y axis in meters for the grid
    # Justin -z_bounds: list[float] = z_bounds, the bounds of the z axis in meters for the grid
    """
    function to get all of the states (waypoints) at whatever fidelity you want, 
    higher better perhaps,
    
    - Justin this requires a lot of parameters, I suggest we refactor to use
    a dictionary and slot in with keyword arguments, but for now this is good enough. 
    FYI the global planner returns waypoints in East North Up (ENU) convention:
        E = x
        N = y 
        Z = z 
        
        ^ N(y)
        |
        |
        ----> E(x)

        z is looking at you by hand right rule
    """
    
    # Justin - You're going to need to instantiate the FWAgent and the Grid first 
    # before you can call out the planner
    ## First off instantiate the FWAgent
    start_position = PositionVector(location[0], 
                                    location[1], 
                                    location[2])
    pitch_dg = np.rad2deg(rotation[1])
    yaw_dg   = np.rad2deg(rotation[2])
    
    fw_agent = FWAgent(start_position, pitch_dg, yaw_dg)
    fw_agent.vehicle_constraints(max_climb_angle_dg=max_climb_angle_dg,
                                 max_psi_turn_dg=max_yaw_turn_dg,
                                 leg_m=leg_m)
    
    goal_position = PositionVector(target_location[0], 
                                   target_location[1], 
                                   target_location[2])

    fw_agent.set_goal_state(goal_position)

    ### Next instantiate the grid
    #this is in case your min bounds are not 0,0,0
    x_max = int(x_bounds[1] - x_bounds[0]) 
    y_max = int(y_bounds[1] - y_bounds[0])
    z_max = int(z_bounds[1] - z_bounds[0])
    grid  = Grid(agent=fw_agent, x_max_m=x_max, y_max_m=y_max,
                    z_max_m=z_max, z_min_m=z_bounds[0])
    
    ### Now you can call the planner
    sparse_astar = SparseAstar(grid=grid, velocity=max_speed)
    sparse_astar.init_nodes() 
    path = sparse_astar.search()
    
    ## Now you can get the waypoints
    return get_waypoint_states(path, max_speed)

model = load_model()

# this is the initial state of the aircraft
states = {
    'u': 20.0,
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

mpc_controller = load_mpc(model, start_state, goal_state)
