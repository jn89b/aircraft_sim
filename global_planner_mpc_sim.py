"""
Example of a global planner for the MPC simulation
"""
import numpy as np
import pandas as pd
import casadi as ca
import pickle as pkl
import matplotlib.pyplot as plt
import math as m


### Guidance MPC ###
from src.Utils import read_lon_matrices, get_airplane_params
from src.aircraft.AircraftDynamics import LinearizedAircraft, LinearizedAircraftCasadi
from src.mpc.FixedWingMPC import LinearizedAircraftMPC
from src.math_lib.VectorOperations import euler_dcm_body_to_inertial, euler_dcm_inertial_to_body, \
    compute_B_matrix    
from src.aircraft.AircraftDynamics import add_noise_to_linear_states

### Global Planner ###
from src.guidance_lib.src.PositionVector import PositionVector
from src.guidance_lib.src.Grid import Grid, FWAgent
from src.guidance_lib.src.Radar import Radar
from src.guidance_lib.src.Obstacle import Obstacle
from src.guidance_lib.src.SparseAstar import SparseAstar


class State:
    def __init__(self, 
                 position:list, 
                 rotation:list, 
                 velocity:list) -> None:
        
        self.position = position
        self.rotation = rotation
        self.velocity = velocity


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

######      Now let's call out the global planner ######## 
#load simple sim
#start_position = PositionVector(,10,0)
start_position = PositionVector(start_state[5], start_state[11], start_state[4])
goal_position = PositionVector(400,450,start_state[4]+40)
fw_agent_psi_dg = 30
fw_agent = FWAgent(start_position, 0, fw_agent_psi_dg)
fw_agent.vehicle_constraints(horizontal_min_radius_m=60, 
                                max_climb_angle_dg=20,
                                max_psi_turn_dg=25)
fw_agent.leg_m = 30

fw_agent.set_goal_state(goal_position)

## create grid
x_max = 500
y_max = 500
grid = Grid(fw_agent, x_max, y_max, 100, 5, 5, 0)

# obs_positions = [(40,60,10)]

#set random seed
num_obstacles = 10
np.random.seed(1)
obs_positions = []
for i in range(num_obstacles):
    x = np.random.randint(150, 350)
    y = np.random.randint(150, 350)
    z = np.random.randint(0, 100)

    #check if obstacle within 50m of start or goal
    if np.linalg.norm(np.array([x,y,z]) - np.array([
            start_position.x, start_position.y, start_position.z])) < 50:
        continue

    obs_positions.append((x,y,z))

obs_list = []
for pos in obs_positions:
    obs_position = PositionVector(pos[0], pos[1], pos[2])
    radius_obs_m = np.random.randint(30, 40)
    some_obstacle = Obstacle(obs_position, radius_obs_m)
    obs_list.append(some_obstacle)
    grid.insert_obstacles(some_obstacle)

sparse_astar = SparseAstar(grid)
sparse_astar.init_nodes()
path = sparse_astar.search()

waypoints = []
for i in range(len(path)):
    x = path[i][0]
    y = path[i][1]
    z = path[i][2]
    position = [x, y, z]
    pitch_rad = np.deg2rad(path[i][3])
    roll_rad = np.deg2rad(path[i][4])
    yaw_rad = np.deg2rad(path[i][5])
    rotation = [pitch_rad, roll_rad, yaw_rad]
    inertial_vel = [25, 0 , 0]
    state = State(position, rotation, inertial_vel)
    waypoints.append(state)
    
distance_tolerance = 10
approach_tolerance = 15

rest_of_waypoints = waypoints[1:]
current_waypoint  = waypoints[0]

current_controls = np.array([controls['delta_e'],
                            controls['delta_t'],
                            controls['delta_a'],
                            controls['delta_r']])
current_state_history = []

x_ref = 0
y_ref = 0
z_ref = 0

counter = 0
counter_break = 1000
idx_start = 1
velocity = 15

for wp in rest_of_waypoints:
    
    goal_x = wp.position[0]
    goal_y = wp.position[1]
    goal_z = wp.position[2]
    goal_psi = wp.rotation[2]
    dx     = goal_x - start_state[5]
    dy     = goal_y - start_state[11] 
    dz     = goal_z - start_state[4]
    lateral_distance = np.sqrt(dx**2 + dy**2)
    
    new_vel = velocity
    new_w = 0.0
    new_q = 0.0
    new_theta = np.arctan2(-dz, np.sqrt(dx**2 + dy**2))
    # new_x = 250
    new_v = 0.0
    new_p = 0.0     
    new_r = 0.0
    new_phi = np.deg2rad(0.0)
    new_y = 0.0
    
    error = np.sqrt(dx**2 + dy**2 + dz**2)
        
    if error >= approach_tolerance:
        new_psi = np.arctan2(dy,dx)
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
    
    print("Moving to the next waypoint")
        
    while error >= distance_tolerance:
        goal_x = wp.position[0]
        goal_y = wp.position[1]
        goal_z = wp.position[2]
        dx     = goal_x - start_state[5]
        dy     = goal_y - start_state[11] 
        dz     = goal_z - start_state[4]
        lateral_distance = np.sqrt(dx**2 + dy**2)
        
        new_vel = velocity
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
        print("Error: ", error)
        if error <= distance_tolerance:
            print("Moving to the next waypoint")
            break            
        
        if error >= approach_tolerance:
            new_psi = np.arctan2(dy,dx)
            new_theta = np.arctan2(-dz, lateral_distance)
        else:
            print("Approaching the waypoint")
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
        
        error = np.sqrt(dx**2 + dy**2 + dz**2)
        
        lin_mpc.reinitStartGoal(start_state, goal_state)
        control_results, state_results = lin_mpc.solveMPCRealTimeStatic(
            start_state, goal_state, current_controls)
        
        control_results = lin_mpc.unpack_controls(control_results)
        state_results = lin_mpc.unpack_states(state_results)
        
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
        inertial_pos = inertial_vel * mpc_params['dt_val']   
        
        x_ref = x_ref + inertial_pos[0]
        y_ref = y_ref + inertial_pos[1]
        z_ref = z_ref + inertial_pos[2]
        
        start_state[5]  =  x_ref + inertial_pos[0]
        start_state[11] = y_ref + inertial_pos[1]
        start_state[4]  =  z_ref + inertial_pos[2]
        
        current_state_history.append(start_state)
        counter += 1
        
        if counter >= counter_break:
            break
        
    if counter >= counter_break:
        print("Breaking the loop")
        break
        
#%%
#plot the position and states 3D position 
fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection':'3d'})

#plot waypoints with direction vectors
for wp in waypoints:
    ax.scatter(wp.position[0], wp.position[1], wp.position[2], c='r', label='waypoints')
    ax.quiver(wp.position[0], wp.position[1], wp.position[2], 
              np.cos(wp.rotation[2]), np.sin(wp.rotation[2]), 0, length=10)

for state in current_state_history:
    ax.scatter(state[5], state[11], state[4], c='b', label='states')

#set z limits 
ax.set_zlim(-20, 20)    

plt.show()