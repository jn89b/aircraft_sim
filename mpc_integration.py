from dataclasses import dataclass


# Justin - Load the SAS planner
from src.guidance_lib.src.SparseAstar import SparseAstar
# Justin - to get the waypoints you're going to need to load the following classes
from src.guidance_lib.src.Grid import Grid, FWAgent
# Justin - If you want to load obstacles, you need to load the obstacle class in here
from src.guidance_lib.src.Obstacle import Obstacle
# I have my own position vector class I use for the grid
from src.guidance_lib.src.PositionVector import PositionVector

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
        

def update(
    location: list[float],  # x, y, z where y is up, in meters
    rotation: list[float],  # roll, pitch, yaw in radians
    velocity: list[float],  # meters per second
    target_location: list[float],
    dt: float,  # seconds
) -> State:
    """afaict, the vectors need to be passed in as normal python objects from C#, 
    we can convert them to cleaner objects if you want, but it's not neccessary, 
    the output though CAN be an object,and I can handle that on the C# side"""
    # location: Vector3 = Vector3(location)
    # rotation: Rotation = Rotation(rotation)
    # velocity: Vector3 = Vector3(velocity)
    # target_location: Vector3 = Vector3(target_location)

    
    pass


def load(
    location: list[float],
    rotation: list[float],
    velocity: list[float],
    target_location: list[float],
    max_speed: float,
    max_yaw_turn_dg: float = 25,
    max_climb_angle_dg: float = 10,
    leg_m:float = 25,
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

    ## Now you can call the planner    
    goal_position = PositionVector(target_location[0], 
                                   target_location[1], 
                                   target_location[2])

    fw_agent.set_goal_state(goal_position)

    ### Next instantiate the grid
    #this is in case you min bounds are not 0,0,0
    x_max = int(x_bounds[1] - x_bounds[0]) 
    y_max = int(y_bounds[1] - y_bounds[0])
    z_max = int(z_bounds[1] - z_bounds[0])
    grid  = Grid(agent=fw_agent, x_max_m=x_max, y_max_m=y_max,
                    z_max_m=z_max, z_min_m=z_bounds[0])
    
    
    ### Now you can call the planner
    sparse_astar = SparseAstar(grid=grid, velocity=max_speed)
    sparse_astar.init_nodes()
    
    # This returns a path 
    path = sparse_astar.search()
    print(path)
    waypoint_states = []
    for i in range(len(path)):
        x = path[0][i]
        y = path[1][i]
        z = path[2][i]
        pitch_rad = deg_to_rad(path[3][i])
        roll_rad = deg_to_rad(path[4][i])
        yaw_rad = deg_to_rad(path[5][i])
        
        location = [x, y, z]
        rotation = [roll_rad, pitch_rad, yaw_rad]
        #I don't have time to take the unit vector for the airspeed and break it into i, j, k
        velocity = [max_speed, 0, 0]
        waypoint_states.append(State(location, rotation, velocity))
    return waypoint_states


if __name__ == "__main__":
    
    starting_location: list[float] = [0, 0, 0]  # x, y, z where y is up, in meters
    starting_rotation: list[float] = [0, 0, 0]  # roll, pitch, yaw in radians
    starting_velocity: list[float] = [0, 0, 0]  # meters per second
    target_location: list[float] = [100, 100, 100]  # meters
    max_speed: float = 30  # meters per second

    ## Justin - To load the waypoints you're going to need to call the SAS function
    ## This has a lot more parameters refer to the function to see what you can do
    states: list[State] = load(
        location=starting_location,
        rotation=starting_rotation,
        velocity=starting_velocity,
        target_location=target_location,
        max_speed=30,
    )
    """
    the above isn't absolutely necessary, 
    but you mentioned this as an option, if the above can be 
    done at a 0.001 deltatime, then i don't see why I can't just store all of those waypoints, 
    buuuuut it would be nice to just be able to call an update function as needed every tick
    
    Justin - The global planner runs at a slower frequnency, so you won't be able to call 
    the update function every tick
    """
    
    """ ideal implmentation """
    location = starting_location
    rotation = starting_rotation
    velocity = starting_velocity
    
    while True:
        state: State = update(
            location=location,
            rotation=rotation,
            velocity=velocity,
            target_location=target_location,
            dt=0.001,
        )
        location = state.location
        rotation = state.rotation
        velocity = state.velocity
        if location == target_location:
            break
