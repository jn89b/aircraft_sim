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

def rad_to_deg(rad: float) -> float:
    return rad * (180 / 3.14159)

def deg_to_rad(deg: float) -> float:
    return deg * (3.14159 / 180)

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
    states = np.array([linearized_state[5], #x   
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
    
    location = [states[0], states[1], -states[2]]
    rotation = [states[6], states[7], -states[10]]
    velocity = [states[3], states[4], states[5]]
    
    return State(location, rotation, velocity)
    

        
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

    # this are the constrints we give to our controller
    # you can make this a parameter if you want 
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
        'v_min': 15, #don't need this really
        'v_max': 35,  #don't need this really
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
    location: list[float],  # x, y, z where y is up, in meters
    rotation: list[float],  # roll, pitch, yaw in radians
    velocity: list[float],  # meters per second
    target_location: list[float],
    dt: float,  # seconds
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
    pitch_dg = rad_to_deg(rotation[1])
    yaw_dg   = rad_to_deg(rotation[2])
    
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
    
    starting_location: list[float] = [20, 20, 0]  # x, y, z where y is up, in meters
    starting_rotation: list[float] = [0, 0, np.deg2rad(45)]  # roll, pitch, yaw in radians
    starting_velocity: list[float] = [15, 0, 0]  # meters per second 
    target_location: list[float] = [300, 290, 0]  # meters
    max_speed: float = 25  # meters per second

    ## Justin - To load the waypoints you're going to need to call the SAS function
    ## This has a lot more parameters refer to the load() see what you can do
    states: list[State] = load(
        location=starting_location,
        rotation=starting_rotation,# 
        velocity=starting_velocity,
        target_location=target_location,
        max_speed=15,
        leg_m=40,
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
    
    distance_tolerance = 10.0 #meters
    approach_tolerance = 25.0
    
    #we want to skip the first waypoint since this is our current state
    rest_of_states    = states[2:]
    current_state     = states[0]
    current_lin_state = convert_state_to_linearized(states[0])
    
    print("current state: ", current_lin_state)
    #set this to an initial value
    current_controls = np.array([0,
                                 0.1,
                                 0.0,
                                 0])
    
    current_state_history = []
    counter = 0
    counter_break = 100

    x_original = current_state.location[0]
    y_original = current_state.location[1]
    z_original = current_state.location[2]
        
    # Load the waypoints and test if this shit works 
    planner_states = pd.read_csv("planner_states.csv")

    #recompute the planner states heading, I made a mistake in the planner
    #The heading should be the heading to the next waypoint  

    states = []
    for i, wp in planner_states.iterrows():
        if i == 0:
            continue
        else:
            dx = wp['x'] - planner_states.iloc[i-1]['x']
            dy = wp['y'] - planner_states.iloc[i-1]['y']
            planner_states.iloc[i-1]['psi_dg'] = np.rad2deg(np.arctan2(dy,dx))
        
        location = [wp['x'], wp['y'], wp['z']]
        orientation = [np.deg2rad(wp['phi_dg']), np.deg2rad(wp['theta_dg']), np.deg2rad(wp['psi_dg'])]
        velocity = [max_speed, 0, 0]
        s = State(location, orientation, velocity)
        states.append(s)
        
    rest_of_states = states[2:]
    current_state = states[0]

        # planner_states.iloc[i] = wp

    for wp in rest_of_states:
        
        distance_from_wp = np.sqrt((wp.location[0]-   current_state.location[0])**2 +
                                    (wp.location[1] - current_state.location[1])**2 +
                                    (wp.location[2] - current_state.location[2])**2)
         
        approach_tolerance = 15.0 
        dt = 0.05
        # linearized_wp = convert_state_to_linearized(wp)    
        print("going to waypoint: ", wp.location)
        print("\n")
        
        while distance_from_wp >= distance_tolerance:
            goal_x = wp.location[0]
            goal_y = wp.location[1]
            goal_h = wp.location[2]    
            dy = goal_y - current_lin_state[11]
            dx = goal_x - current_lin_state[5]
            dz = goal_h - current_lin_state[4]
            new_vel = 15.0
            new_w = 0.0
            new_q = 0.0
            new_theta = np.arctan2(-dz, np.sqrt(dx**2 + dy**2))
            new_height = goal_h
            # new_x = 250
            new_v = 0.0
            new_p = 0.0     
            new_r = 0.0
            new_phi = np.deg2rad(0.0)
            new_y = 0.0
            
            if distance_from_wp >= approach_tolerance:
                new_psi = np.arctan2(dy,dx)
            # else:
            #     new_psi = wp.rotation[2]
                
            print("desired psi: ", np.rad2deg(new_psi))
            print("current psi: ", np.rad2deg(current_lin_state[10]))
                
            goal_state = np.array([new_vel,
                                    new_w,
                                    new_q,
                                    new_theta,
                                    goal_h,
                                    goal_x,
                                    new_v,
                                    new_p,
                                    new_r,
                                    new_phi,
                                    new_psi,
                                    goal_y])
        
            aircraft_mpc.reinitStartGoal(current_lin_state, goal_state)
            
            control_results, state_results = aircraft_mpc.solveMPCRealTimeStatic(
                current_lin_state, goal_state, current_controls)
            
            #unpack the results
            control_results = aircraft_mpc.unpack_controls(control_results)
            state_results = aircraft_mpc.unpack_states(state_results)
            idx_start = 1
            current_lin_state = np.array([state_results['u'][idx_start],
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
            
            current_controls = np.array([control_results['delta_e'][idx_start],
                                        control_results['delta_t'][idx_start],
                                        control_results['delta_a'][idx_start],
                                        control_results['delta_r'][idx_start]])
                
            R = euler_dcm_body_to_inertial(state_results['phi'][idx_start],
                                        state_results['theta'][idx_start],
                                        state_results['psi'][idx_start])
            
            body_vel = np.array([state_results['u'][idx_start],
                                state_results['v'][idx_start],
                                state_results['w'][idx_start]])
            
            inertial_vel = np.matmul(R, body_vel)
            inertial_pos = inertial_vel * 0.05
            
            x_original = x_original + inertial_pos[0]
            y_original = y_original + inertial_pos[1]
            #z_original = z_original  inertial_pos[2]

            #replace the position with the inertial position
            current_lin_state[5] = x_original
            current_lin_state[11] = y_original
            current_lin_state[4] = z_original
            
            dx = goal_x - current_lin_state[5]
            dy = goal_y - current_lin_state[11]
            dz = goal_h - current_lin_state[4]
            
            distance_from_wp = np.sqrt(dx**2 + dy**2 + dz**2)
            print("Distance from waypoint: ", distance_from_wp)
            
            counter += 1
            if counter > counter_break:
                break
            
            current_state = convert_linearized_to_state(current_lin_state)
            current_state_history.append(current_state)
        
        if counter > counter_break:
            break
            
# plot the current state history
x = [state.location[0] for state in current_state_history]
y = [state.location[1] for state in current_state_history]
z = [state.location[2] for state in current_state_history]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', label='trajectory')
ax.plot(x, y, z, 'o-')

#plot the waypoint
ax.plot([state.location[0] for state in states], 
        [state.location[1] for state in states], 
        [state.location[2] for state in states], 'o-', label='waypoints')

ax.legend()

#plot psi
psi = [state.rotation[2] for state in current_state_history]
fig2, ax2 = plt.subplots()
ax2.plot(np.rad2deg(psi), 'o-')

plt.show()