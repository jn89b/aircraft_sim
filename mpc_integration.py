import numpy as np
import pandas as pd
import casadi as ca
import pickle as pkl
import matplotlib.pyplot as plt
import math as m


from dataclasses import dataclass
### Justin - This is imports needed for the global planner 
#  Load the SAS planner
from src.guidance_lib.src.SparseAstar import SparseAstar
# to get the waypoints you're going to need to load the following classes
from src.guidance_lib.src.Grid import Grid, FWAgent
# If you want to load obstacles, you need to load the obstacle class in here
from src.guidance_lib.src.Obstacle import Obstacle
# I have my own position vector class I use for the grid
from src.guidance_lib.src.PositionVector import PositionVector
from src.math_lib.VectorOperations import euler_dcm_body_to_inertial, euler_dcm_inertial_to_body, \
    compute_B_matrix


### Justin - This is for the Model Predictive Controller, you need to load the following classes
from src.Utils import read_lon_matrices, get_airplane_params
from src.aircraft.AircraftDynamics import LinearizedAircraft, LinearizedAircraftCasadi
from src.mpc.FixedWingMPC import LinearizedAircraftMPC


class Vector3:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __init__(self, vector: list[float]):
        self.x = vector[0]
        self.y = vector[1]
        self.z = vector[2]


class Rotation:
    def __init__(self, roll: float, pitch: float, yaw: float):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def __init__(self, rotation: list[float]):
        self.roll = rotation[0]
        self.pitch = rotation[1]
        self.yaw = rotation[2]


class Controls:
    def __init__(self, elevator: float, throttle: float, aileron: float, rudder: float):
        self.elevator = elevator # in radians
        self.throttle = throttle # in percent 0-1
        self.aileron = aileron   # in radians
        self.rudder = rudder     # in radians

    def __init__(self, control: list[float]):
        self.elevator = control[0]
        self.throttle = control[1]
        self.aileron = control[2]
        self.rudder = control[3]

class State:
    location: list[float]  # x, y, z where y is up, in meters
    rotation: list[float]  # roll, pitch, yaw in radians
    velocity: list[float]  # x, y, z where y is up, in meters per second

    def __init__(self, location: list[float], rotation: list[float], velocity: list[float]):
        self.location = location
        self.rotation = rotation
        self.velocity = velocity
        
def convert_state_to_linearized(state: State, 
                                p:float=0,
                                q:float=0,
                                r:float=0
                                ) -> np.array:
    """
    Helper function:
    This function converts the state of the aircraft to the state space representation
    Since the State class is in the ENU convention, we need to convert it to the NED convention
    """
    # Convert ENU position to NED position
    enu_x = state.location[0]
    enu_y = state.location[1]
    enu_z = state.location[2]
    ned_x = enu_x
    ned_y = enu_y
    ned_z = enu_z
    
    # Convert ENU rotation to NED rotation        
    enu_roll  = state.rotation[0]
    enu_pitch = state.rotation[1]
    enu_yaw   = state.rotation[2]
    phi       = enu_roll
    theta     = enu_pitch
    psi       =  enu_yaw
    
    # Convert ENU inertial velocity to NED BODY velocity
    enu_u = state.velocity[0]
    enu_v = state.velocity[1]
    enu_w = state.velocity[2]

    ned_u = enu_u
    ned_v = enu_v
    ned_w = enu_w

    # R = euler_dcm_inertial_to_body(phi, theta, psi)
    # ned_u, ned_v, ned_w = np.dot(R, np.array([enu_u, enu_v, enu_w]))
    
    # we need to return this array in the proper order to feed to the MPC
    linearized_array = np.array([ned_u, #u 
                                 ned_w, #w
                                 q,     #q
                                 theta, #theta
                                 ned_z, #h
                                 ned_x, #x
                                 ned_v, #v
                                 p,     #p
                                 r,     #r
                                 phi,     #phi
                                 psi,   #psi
                                 ned_y])
    
    return linearized_array
    
        
def convert_linearized_to_state(linearized_state: np.array) -> State:
    """ 
    Helper function:
    This function converts the state space representation to the state of the aircraft
    Since the State class is in the ENU convention, we need to convert it to the NED convention
    """
    sim_states = np.array([linearized_state[5], #x   
                     linearized_state[11], #y 
                     linearized_state[4], #z
                     linearized_state[0], #u
                     linearized_state[6], #v
                     linearized_state[1], #w
                     linearized_state[9], #phi
                     linearized_state[3], #theta
                     linearized_state[10], #psi
                     linearized_state[7], #p
                     linearized_state[2], #q
                     linearized_state[8]]) #r

    sim_loc = [sim_states[0], sim_states[1], sim_states[2]]
    sim_rot = [sim_states[6], sim_states[7], sim_states[8]]
    sim_vel = [sim_states[3], sim_states[4], states[5]]
    
    return State(sim_loc, sim_rot, sim_vel)
           
def load_linearized_model() -> LinearizedAircraftCasadi:
    """
    This loads the linearized model of the aircraft for use in the MPC
    Justin - First off we need to load the airplane parameter coefficients as well 
    as the linearized matrices 
    """
    df = pd.read_csv("SIM_Plane_h_vals.csv")
    airplane_params = get_airplane_params(df)
    with open('A_full.pkl', 'rb') as f:
        A_full = pkl.load(f)

    with open('B_full.pkl', 'rb') as f:
        B_full = pkl.load(f)

    aircraft_linear = LinearizedAircraftCasadi(airplane_params, 
                                               A_full, 
                                               B_full)

    return aircraft_linear

def load_mpc(aircraft_model:LinearizedAircraft, 
             start_state:State, #
             goal_state:State,
             dt_val:float = 0.05,
             N:int = 15) -> LinearizedAircraftMPC:
    # weighting matrices for state the higher the value the more the 
    # state is penalized (we care more about it the higher it is) 
    # you can make this a parameter if you want
    #weighting matrices for state
    Q = np.diag([
        0.75, #u
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
    # you can make this a parameter if you want
    r_control = 0.4
    R = np.diag([
        r_control, #delta_e
        1.0, #delta_t
        r_control, #delta_a
        r_control, #delta_r
    ])

    mpc_params = {
        'model': aircraft_model, # the linearized aircraft model
        'dt_val': dt_val, #time step for the MPC
        'N': N, #prediction horizon
        'Q': Q, #state weighting matrix
        'R': R, #control weighting matrix
    }

    # these are the state and control constraints we give to our controller
    # you can make this a parameter if you want, I don't recomend you do this
    # since the constraints are specific to the aircraft
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
        'v_min': -35, 
        'v_max': 35,  
        'p_min': np.deg2rad(-60),
        'p_max': np.deg2rad(60),
        'r_min': np.deg2rad(-60),
        'r_max': np.deg2rad(60),
        'phi_min': np.deg2rad(-45),
        'phi_max': np.deg2rad(45)
    }

    linearized_start = convert_state_to_linearized(start_state)
    linearized_goal  = convert_state_to_linearized(goal_state)
    # now lets create the MPC controller
    # you don't need to call the following again once you have created the controller
    lin_mpc = LinearizedAircraftMPC(mpc_params, lin_mpc_constraints)
    lin_mpc.initDecisionVariables()
    lin_mpc.reinitStartGoal(linearized_start, linearized_goal)
    lin_mpc.computeCost()
    lin_mpc.defineBoundaryConstraints()
    lin_mpc.addAdditionalConstraints()

    return lin_mpc

def update(
    location: list[float],   # x, y, z where y is up, in meters
    rotation: list[float],   # roll, pitch, yaw in radians
    velocity: list[float],   # meters per second
    input_control:Controls, # the current control input
    target_state: State,     # the waypoint as a state
    mpc: LinearizedAircraftMPC,
    approach_tol_m: float = 20,
    idx_start: int = 1, #idx where the start state is in the trajectory
    dt: float =0.05,  # seconds
    
) -> tuple:
    """
    afaict, the vectors need to be passed in as normal python objects from C#, 
    we can convert them to cleaner objects if you want, but it's not neccessary, 
    the output though CAN be an object,and I can handle that on the C# side
    """
    # location: Vector3 = Vector3(location)
    # rotation: Rotation = Rotation(rotation)
    # velocity: Vector3 = Vector3(velocity)
    # target_location: Vector3 = Vector3(target_location) -> the waypoint
    dist_x = target_state.location[0] - location[0]
    dist_y = target_state.location[1] - location[1]
    dist_z = target_state.location[2] - location[2]
    
    lateral_distance = m.sqrt(dist_x**2 + dist_y**2)
    error = np.sqrt(dist_x**2 + dist_y**2 + dist_z**2)
    
    start_state_vec = convert_state_to_linearized(
        State(location, rotation, velocity))
    
    if error>=approach_tol_m:
        target_psi_rad = np.arctan2(dist_y, dist_x)
        target_theta_rad = np.arctan2(-dist_z, lateral_distance)
    else:
        target_psi_rad = target_state.rotation[2]
        
    goal_state_vec = np.array([target_state.velocity[0], #u
                            target_state.velocity[2], #w
                            0.0, #q
                            target_theta_rad, #theta
                            target_state.location[2], #h
                            target_state.location[0], #x
                            target_state.velocity[1], #v
                            0.0, #p
                            0.0, #r
                            target_state.rotation[0], #phi
                            target_psi_rad, #psi
                            target_state.location[1] #y
                            ])
    
    mpc.reinitStartGoal(start_state_vec, goal_state_vec)
    
    current_controls_vec = np.array([input_control.elevator,
                                input_control.throttle,
                                input_control.aileron,
                                input_control.rudder])
    
    u, x = mpc.solveMPCRealTimeStatic(start_state_vec, 
                                      goal_state_vec, 
                                      current_controls_vec)
    
    control_traj = mpc.unpack_controls(u)
    state_traj = mpc.unpack_states(x)

    start_state = np.array([state_traj['u'][idx_start],
                            state_traj['w'][idx_start],
                            state_traj['q'][idx_start],
                            state_traj['theta'][idx_start],
                            state_traj['h'][idx_start],
                            state_traj['x'][idx_start],
                            state_traj['v'][idx_start],
                            state_traj['p'][idx_start],
                            state_traj['r'][idx_start],
                            state_traj['phi'][idx_start],
                            state_traj['psi'][idx_start],
                            state_traj['y'][idx_start]])
    
    control_output_vec = np.array([control_traj['delta_e'][idx_start],
                            control_traj['delta_t'][idx_start],
                            control_traj['delta_a'][idx_start],
                            control_traj['delta_r'][idx_start]])
    
    # Need to go from body to inertial frame position to get 
    # inertial velocity
    R = euler_dcm_body_to_inertial(state_traj['phi'][idx_start],
                                state_traj['theta'][idx_start],
                                state_traj['psi'][idx_start])
    body_vel = np.array([state_traj['u'][idx_start],
                        state_traj['v'][idx_start],
                        state_traj['w'][idx_start]])
                        
    inertial_vel = np.matmul(R, body_vel)
    
    location = [start_state[5], start_state[11], start_state[4]]
    rotation = [start_state[9], start_state[3], start_state[10]]
    velocity = [inertial_vel[0], inertial_vel[1], inertial_vel[2]]
    
    output_state = State(location, rotation, velocity)
    output_control = Controls(control_output_vec)
    
    return output_state, output_control

    
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
    max_climb_angle_dg: float = 15,
    leg_m:float = 30,
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


if __name__ == "__main__":
    
    starting_location: list[float] = [0, 0, -3]  # x, y, z where y is up, in meters
    starting_rotation: list[float] = [0, np.deg2rad(-0.03), np.deg2rad(45)]  # roll, pitch, yaw in radians
    starting_velocity: list[float] = [20, 0, 0]  # meters per second 
    target_location: list[float] = [400, 450, 20]  # meters
    max_speed: float = 25  # meters per second

    ## Justin - To load the waypoints you're going to need to call the SAS function
    ## This has a lot more parameters refer to the load() see what you can do
    states: list[State] = load(
        location=starting_location,
        rotation=starting_rotation,# 
        velocity=starting_velocity,
        target_location=target_location,
        max_speed=15,
        leg_m=30,
    )
    
    ## Justin - You're going to need instantiate the MPC here
    aircraft_model = load_linearized_model()   
    aircraft_mpc  =  load_mpc(aircraft_model,
                            start_state=states[0], 
                            goal_state=states[-1],
                            dt_val=0.05,
                            N=15)
    
    """
    the above isn't absolutely necessary, 
    but you mentioned this as an option, if the above can be 
    done at a 0.001 deltatime, then i don't see why I can't just store all of those waypoints, 
    buuuuut it would be nice to just be able to call an update function as needed every tick
    
    Justin - The global planner runs at a slower frequnency than 0.001s, so you won't be 
    able to call the update function every tick. Just call it once, 
    cache the initial waypoints and then send the waypoints to the MPC controller 
    """
    
    """ ideal implmentation """
    location = starting_location
    rotation = starting_rotation
    velocity = starting_velocity
    
    """
    This is where you would use your update function to update the state of the vehicle
    """
    # while True:
    #     state: State = update(
    #         location=location,
    #         rotation=rotation,
    #         velocity=velocity,
    #         target_location=target_location,
    #         dt=0.001,
    #     )
    #     location = state.location
    #     rotation = state.rotation
    #     velocity = state.velocity
    
    #     #you should have a tolerance check here, but I'm not sure what that would be
    #     # if location == target_location:
    #     #     break

    #we want to skip the first waypoint since this is our current state
    rest_of_states    = states[1:]
    current_state     = states[0]
    current_lin_state = np.array([
        current_state.velocity[0], #u
        current_state.velocity[2], #w
        0.0, #q
        current_state.rotation[1], #theta
        current_state.location[2], #h
        current_state.location[0], #x
        current_state.velocity[1], #v
        0.0, #p
        0.0, #r
        current_state.rotation[0], #phi
        current_state.rotation[2], #psi
        current_state.location[1] #y
    ])
            
    current_state_history = []
    counter = 0
    counter_break = 500
    dt = 0.05

    #these are zeroed out reference whiles since we based on the linearized state
    x_ref = 0
    y_ref = 0
    z_ref = 0
    idx_start = 1
    
    rest_of_waypoints = states[1:]
    current_state  = states[0]
    
    approach_tolerance = 20
    distance_tolerance = 10
    
    start_state = current_lin_state
    
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

    current_controls = np.array([controls['delta_e'],
                                controls['delta_t'],
                                controls['delta_a'],
                                controls['delta_r']])
    
    for wp in rest_of_waypoints:
        
        goal_x = wp.location[0]
        goal_y = wp.location[1]
        goal_z = wp.location[2]
        goal_psi = wp.rotation[2]
        dx     = goal_x - start_state[5]
        dy     = goal_y - start_state[11]
        dz     = goal_z - start_state[4]
        lateral_distance = np.sqrt(dx**2 + dy**2)
        
        error = np.sqrt(dx**2 + dy**2 + dz**2)
            
        if error >= approach_tolerance:
            new_psi = np.arctan2(dy,dx)
        else:
            new_psi = goal_psi
                        
        while error >= distance_tolerance:
            goal_x = wp.location[0]
            goal_y = wp.location[1]
            goal_z = wp.location[2]
            dx     = goal_x - start_state[5]
            dy     = goal_y - start_state[11] 
            dz     = goal_z - start_state[4]
            lateral_distance = np.sqrt(dx**2 + dy**2)
            
            new_vel = 15
            new_w = 0.0
            new_q = 0.0
            new_theta = np.arctan2(-dz, lateral_distance)
            # new_x = 250
            new_v = 0.0
            new_p = 0.0     
            new_r = 0.0
            new_phi = np.deg2rad(0.0)
            new_psi = np.arctan2(dy,dx)
            new_y = 0.0
            
            error = np.sqrt(dx**2 + dy**2 + dz**2)
            
            if error <= distance_tolerance:
                print('reached waypoint')
                break            
            
            if error >= approach_tolerance:
                new_psi = np.arctan2(dy,dx)
                new_theta = np.arctan2(-dz, lateral_distance)
            else:
                new_psi = goal_psi

            goal_state = np.array([new_vel,
                                    new_w,
                                    new_q,
                                    new_theta,
                                    goal_z,
                                    goal_x,
                                    new_v,
                                    new_p,
                                    new_r,
                                    new_phi,
                                    new_psi,
                                    new_y])
            
            # error = np.sqrt(dx**2 + dy**2 + dz**2)
            aircraft_mpc.reinitStartGoal(start_state, goal_state)
            control_results, state_results = aircraft_mpc.solveMPCRealTimeStatic(
                start_state, goal_state, current_controls)
            
            control_results = aircraft_mpc.unpack_controls(control_results)
            state_results = aircraft_mpc.unpack_states(state_results)
            
            start_state = np.array([state_results['u'][idx_start],
                                    state_results['w'][idx_start],
                                    state_results['q'][idx_start],
                                    state_results['theta'][idx_start],
                                    state_results['h'][idx_start],
                                    state_results['x'][idx_start],
                                    state_results['v'][idx_start],
                                    state_results['p'][idx_start],
                                    state_results['r'][idx_start],
                                    state_results['phi'][idx_start],
                                    state_results['psi'][idx_start],
                                    state_results['y'][idx_start]])

            
            start_control = np.array([control_results['delta_e'][idx_start],
                                        control_results['delta_t'][idx_start],
                                        control_results['delta_a'][idx_start],
                                        control_results['delta_r'][idx_start]])
            
            # Need to go from body to inertial frame position
            R = euler_dcm_body_to_inertial(state_results['phi'][idx_start],
                                        state_results['theta'][idx_start],
                                        state_results['psi'][idx_start])
            body_vel = np.array([state_results['u'][idx_start],
                                state_results['v'][idx_start],
                                state_results['w'][idx_start]])

            inertial_vel = np.matmul(R, body_vel)
            inertial_pos = inertial_vel * 0.05
            
            x_ref = x_ref + inertial_pos[0]
            y_ref = y_ref + inertial_pos[1]
            z_ref = z_ref + inertial_pos[2]
            
            location = [start_state[5], start_state[11], start_state[4]]
            rotation = [state_results['phi'][idx_start],
                        state_results['theta'][idx_start],
                        state_results['psi'][idx_start]]
            velocity = [inertial_vel[0], inertial_vel[1], inertial_vel[2]]
            
            sim_state = State(location, rotation, velocity)
            
            current_state_history.append(sim_state)
            counter += 1
            
            if counter >= counter_break:
                break
            
        if counter>=counter_break:
            break

            
# plot the current state history
x = [state.location[0] for state in current_state_history]
y = [state.location[1] for state in current_state_history]
z = [state.location[2] for state in current_state_history]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', label='trajectory')
ax.plot(x, y, z, 'o-')

#plot the waypoint
ax.plot([state.location[0] for state in rest_of_states], 
        [state.location[1] for state in rest_of_states], 
        [state.location[2] for state in rest_of_states], 'o-', label='waypoints')

ax.legend()

#plot psi
psi = [state.rotation[2] for state in current_state_history]
fig2, ax2 = plt.subplots()
ax2.plot(np.rad2deg(psi), 'o-')

plt.show()