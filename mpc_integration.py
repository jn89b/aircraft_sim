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

class Rotation:
    def __init__(self, roll: float, pitch: float, yaw: float):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
            

class Controls:
    def __init__(self, elevator: float, throttle: float, aileron: float, rudder: float):
        self.elevator = elevator # in radians
        self.throttle = throttle # in percent 0-1
        self.aileron = aileron   # in radians
        self.rudder = rudder     # in radians

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

    loc = [linearized_state[5], linearized_state[11], linearized_state[4]]
    rot = [linearized_state[9], linearized_state[3], linearized_state[10]]
    vel = [linearized_state[0], linearized_state[6], linearized_state[1]]
    
    return State(loc, rot, vel)
           

def compute_distance3D(start: list[float], end: list[float]) -> float:
    """
    Helper function:
    This function computes the distance between two points in 3D space
    """
    return np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2 + (end[2] - start[2])**2)
        
def compute_distance2D(start: list[float], end: list[float]) -> float:
    """
    Helper function:
    This function computes the distance between two points in 2D space
    """
    return np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)



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

def get_linear_goal_vector(waypoint_state:State,
                           current_location:list[float],
                           approach_tolerance_m:float=15) -> np.array:
    lat_distance = compute_distance2D(waypoint_state.location, 
                                          current_location)
    dz = waypoint_state.location[2] - current_location[2]
    goal_vel = 15
    goal_w = 0.0
    goal_q = 0.0
    goal_theta = np.arctan2(-dz, lat_distance)
    
    if lat_distance >= approach_tolerance_m:
        goal_psi = np.arctan2(waypoint_state.location[1] - current_location[1],
                              waypoint_state.location[0] - current_location[0])
    else:
        goal_psi = waypoint_state.rotation[2]
        
    lin_goal_state = np.array([goal_vel, #u
                            goal_w, #w
                            goal_q, #q
                            goal_theta, #theta
                            waypoint_state.location[2], #h
                            waypoint_state.location[0], #x
                            0.0, #v
                            0.0, #p
                            0.0, #r
                            0.0, #phi
                            goal_psi, #psi
                            waypoint_state.location[1]]) #y
    
    return lin_goal_state

def get_trajectory(input_state_vector: np.array,
                   input_control_vector: np.array,
                   goal_state_vector: np.array,
                   mpc_controller:LinearizedAircraftMPC) -> tuple:
    
    mpc_controller.reinitStartGoal(input_state_vector, input_control_vector)
    U, X = mpc_controller.solveMPCRealTimeStatic(input_state_vector, 
                                                 goal_state_vector,
                                                 input_control_vector)
    u_traj = mpc_controller.unpack_controls(U)
    x_traj   = mpc_controller.unpack_states(X)
    
    return x_traj, u_traj
    
    
def get_next_state(state_traj_results:dict,
                   control_traj_results:dict,
                   idx_start:int) -> tuple:
    
    state_vec = np.array([state_traj_results['u'][idx_start],
                        state_traj_results['w'][idx_start],
                        state_traj_results['q'][idx_start],
                        state_traj_results['theta'][idx_start],
                        state_traj_results['h'][idx_start],
                        state_traj_results['x'][idx_start],
                        state_traj_results['v'][idx_start],
                        state_traj_results['p'][idx_start],
                        state_traj_results['r'][idx_start],
                        state_traj_results['phi'][idx_start],
                        state_traj_results['psi'][idx_start],
                        state_traj_results['y'][idx_start]])
    
    control_vec = np.array([control_traj_results['delta_e'][idx_start],
                        control_traj_results['delta_t'][idx_start],
                        control_traj_results['delta_a'][idx_start],
                        control_traj_results['delta_r'][idx_start]])
    
    
    # R = euler_dcm_body_to_inertial(state_traj_results['phi'][idx_start],
    #                             state_traj_results['theta'][idx_start],
    #                             state_traj_results['psi'][idx_start])
    # body_vel = np.array([state_traj_results['u'][idx_start],
    #                     state_traj_results['v'][idx_start],
    #                     state_traj_results['w'][idx_start]])
    
    # inertial_vel = np.matmul(R, body_vel)
    
    # location = [state_vec[5], state_vec[11], state_vec[4]]
    # rotation = [state_vec[9], state_vec[3], state_vec[10]]
    # inertial_vel = [inertial_vel[0], inertial_vel[1], inertial_vel[2]]
    
    # state = State(location, rotation, inertial_vel)
    
    return state_vec, control_vec

def update(
    start_location: list[float],   # x, y, z where y is up, in meters
    start_rotation: list[float],   # roll, pitch, yaw in radians
    start_velocity: list[float],   # meters per second
    input_controls: list[float],   # in radians and percent
    target_location: list[float],  # meters
    target_rotation: list[float],  # roll, pitch, yaw in radians
    mpc: LinearizedAircraftMPC, 
    approach_tol_m: float = 20,
    idx_start: int = 1, #idx where the start state is in the trajectory
    dt: float =0.05,  # seconds
) -> State:
    """
    afaict, the vectors need to be passed in as normal python objects from C#, 
    we can convert them to cleaner objects if you want, but it's not neccessary, 
    the output though CAN be an object,and I can handle that on the C# side
    """
    # location: Vector3 = Vector3(location)
    # rotation: Rotation = Rotation(rotation)
    # velocity: Vector3 = Vector3(velocity)
    # target_location: Vector3 = Vector3(target_location) -> the waypoint
    lin_start_vec = convert_state_to_linearized(State(start_location,
                                                        start_rotation,
                                                        start_velocity))
    
    distance_from_wp = compute_distance3D(start_location, target_location)
    
    dx = target_location[0] - start_location[0]
    dy = target_location[1] - start_location[1]
    dz = target_location[2] - start_location[2]
    target_theta = np.arctan2(-dz, compute_distance2D(target_location, start_location))    
    
    if distance_from_wp >= approach_tol_m:
        target_psi = np.arctan2(dy,dx)
    else:
        target_psi = target_rotation[2]

        
    min_vel = 15
    lin_goal_vec = np.array([min_vel, #set this to the minimum speed u
                            0.0, #w
                            0.0, #q
                            target_theta, #theta
                            target_location[2], #h
                            target_location[0], #x
                            0.0, #v
                            0.0, #p
                            0.0, #r
                            0.0, #phi
                            target_psi, #psi
                            0]) #y
    
    mpc.reinitStartGoal(lin_start_vec, lin_goal_vec)
    U, X = mpc.solveMPCRealTimeStatic(
        lin_start_vec, lin_goal_vec, input_controls)
    
    control_traj = mpc.unpack_controls(U)
    state_traj   = mpc.unpack_states(X)
    
    output_state = np.array([state_traj['u'][idx_start],
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
    
    output_control = np.array([control_traj['delta_e'][idx_start],
                            control_traj['delta_t'][idx_start],
                            control_traj['delta_a'][idx_start],
                            control_traj['delta_r'][idx_start]])
    
    # R = euler_dcm_body_to_inertial(state_traj['phi'][idx_start],
    #                             state_traj['theta'][idx_start],
    #                             state_traj['psi'][idx_start])
    # body_vel = np.array([state_traj['u'][idx_start],
    #                     state_traj['v'][idx_start],
    #                     state_traj['w'][idx_start]])
                        
    # inertial_vel = np.matmul(R, body_vel)
    
    # output_location = [output_state[5], output_state[11], output_state[4]]
    # output_rotation = [output_state[9], output_state[3], output_state[10]]
    # output_velocity = [inertial_vel[0], inertial_vel[1], inertial_vel[2]]

    # output_state = State(output_location, output_rotation, output_velocity)
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
    target_location:   list[float] = [400, 450, -3] # meters
    starting_controls: list[float] = [0,   #aileron in radians
                                      0.1, #throttle in percent
                                      0,   #elevator in radians
                                      0]   #rudder in radians
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
    controls = starting_controls 
    
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

    current_state_history = []
    approach_tolerance = 20
    distance_tolerance = 10
    
    counter = 0
    counter_break = 200
    dt = 0.05
    idx_start = 1
    
    #skip the first waypoint since this is the current state    
    current_state  = states[0]
    rest_of_waypoints = states[1:]
    
    ### Note this has to be a numpy array
    start_state = convert_state_to_linearized(current_state)
    controls = np.array(controls)
  
    for wp in rest_of_waypoints:
        
        lateral_distance = compute_distance2D(wp.location, location)
        error = compute_distance3D(wp.location, location)
        
        while error >= distance_tolerance:
        
            error = compute_distance3D(wp.location, 
                                       location)
    
            if error <= distance_tolerance:
                print('reached waypoint')
                break
            
            goal_state = get_linear_goal_vector(wp,
                                                location,
                                                approach_tolerance)

            state_results, control_results = get_trajectory(start_state, 
                                                           controls,
                                                           goal_state, 
                                                           aircraft_mpc)

            start_state, controls = get_next_state(state_results,
                                                            control_results,
                                                            idx_start)
            
            sim_state = convert_linearized_to_state(start_state)
            location = sim_state.location
            rotation = sim_state.rotation
            velocity = sim_state.velocity
            
            current_state_history.append(sim_state)

            counter+=1
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
ax.plot([state.location[0] for state in rest_of_waypoints], 
        [state.location[1] for state in rest_of_waypoints], 
        [state.location[2] for state in rest_of_waypoints], 'o-', label='waypoints')

ax.legend()

#plot psi
psi = [state.rotation[2] for state in current_state_history]
fig2, ax2 = plt.subplots()
ax2.plot(np.rad2deg(psi), 'o-')

plt.show()