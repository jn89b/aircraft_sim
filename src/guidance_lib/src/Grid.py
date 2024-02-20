import numpy as np
import math as m
from src.guidance_lib.src.PositionVector import PositionVector
from src.guidance_lib.src.Obstacle import Obstacle


class FWAgent():
    
    """
    Represents an agent (e.g., aircraft) with position, orientation, and movement constraints.
    
    Attributes:
        position (PositionVector): Current position of the agent.
        theta_dg (float): Pitch angle in degrees.
        psi_dg (float): Azimuth heading in degrees.
        radius_m (float): Radius of the agent in meters.
        leg_m (float): Leg length of the agent's path in meters.
    
    Methods:
        __init__: Constructor to initialize the agent.
        set_current_state: Updates the state of the agent.
        set_goal_state: Sets a goal position for the agent.
        vehicle_constraints: Sets the vehicle constraints.
        get_moves: Calculates all possible forward moves based on current position and heading.
    """
    
    
    def __init__(self, position:PositionVector, 
                 theta_dg:float=0, psi_dg:float=0, leg_m:float=50) -> None:
        self.position = position
        self.theta_dg = theta_dg #pitch anglemoves.append([next_x, next_y, next_z])
        self.psi_dg = psi_dg # this is azmith heading
        self.radius_m = 5 #radius of aircraft meters
        self.leg_m = leg_m #leg length in meters

    def set_current_state(self, position:PositionVector, 
                  theta_dg:float=0, psi_dg:float=0) -> None:
        """
        update state of aircraft
        """
        self.position = position
        self.theta_dg = theta_dg
        self.psi_dg = psi_dg

    def set_goal_state(self, position:PositionVector) -> None:
        self.goal_position = position

    def vehicle_constraints(self, horizontal_min_radius_m:float=35, 
                            max_climb_angle_dg:float=10, 
                            max_psi_turn_dg:float=45,
                            leg_m:float=25) -> None:
        """
        horizontal_min_turn = v^2/ (g * tan(phi)) where theta is bank angle
        """
        self.horizontal_min_radius_m = horizontal_min_radius_m
        self.max_climb_angle_dg = max_climb_angle_dg
        self.max_psi_turn_dg = max_psi_turn_dg
        self.leg_m = leg_m

    #this can be sped up 
    def get_moves(self, position:PositionVector, curr_psi_dg:float,
                  current_theta_dg:float, step_psi=5) -> list:
        """
        based on current position and heading get all 
        possible forward moves
        """
        
        moves = []
        ac_max_psi_dg = self.max_psi_turn_dg
        ac_max_theta_dg = self.max_climb_angle_dg

        max_z = round(self.leg_m*np.sin(np.deg2rad(ac_max_theta_dg)))
        min_z = round(self.leg_m*np.sin(np.deg2rad(-ac_max_theta_dg)))
        step_z = 5

        #round to nearest 5
        max_z = max_z + (5 - max_z) % 5
        min_z = min_z + (5 - min_z) % 5

        for i in range(0,ac_max_psi_dg+step_psi, step_psi):
            next_psi_dg = curr_psi_dg + i
            if next_psi_dg > 360:
                next_psi_dg -= 360
            if next_psi_dg < 0:
                next_psi_dg += 360

            psi_rad = np.deg2rad(next_psi_dg)
            next_x = position.x + round(self.leg_m*(np.cos(psi_rad)))
            next_y = position.y + round(self.leg_m*(np.sin(psi_rad)))
            
            for theta in range(-ac_max_theta_dg, ac_max_theta_dg+step_psi, step_psi):
                next_theta_dg = current_theta_dg + theta
                
                if next_theta_dg > 360:
                    next_theta_dg -= 360
                if next_theta_dg < 0:
                    next_theta_dg += 360
                    
                theta_rad = np.deg2rad(next_theta_dg)

            #     x = round(self.leg_m*(np.sin(theta_rad))*(np.cos(psi_rad)))
            #     y = round(self.leg_m*(np.sin(theta_rad))*(np.sin(psi_rad)))
            #     z = round(self.leg_m*(np.cos(theta_rad)))
                
            #     next_x = position.x + x
            #     next_y = position.y + y
            #     next_z = position.z + z
            #     moves.append([next_x, next_y, next_z])
                # next_x = position.x + round(self.leg_m*(np.cos(psi_rad)))    
                # next_x = position.x + round(self.leg_m*(np.cos(psi_rad)))
                # next_y = position.y + round(self.leg_m*(np.sin(psi_rad)))
                for z in range(min_z, max_z+step_z, step_z):
                    next_z = position.z + z
                    moves.append([next_x, next_y, next_z])
            
            # next_x = position.x + round(self.leg_m*(np.cos(psi_rad)))
            # next_y = position.y + round(self.leg_m*(np.sin(psi_rad)))
            # #this should not be z but the steps based on your climb angle
            
            # for z in range(min_z, max_z+step_z, step_z):
            #     next_z = position.z + z
            #     moves.append([next_x, next_y, next_z])

        for i in range(0,ac_max_psi_dg+step_psi, step_psi):
            next_psi_dg = curr_psi_dg - i
            if next_psi_dg > 360:
                next_psi_dg -= 360
            if next_psi_dg < 0:
                next_psi_dg += 360        

            psi_rad = np.deg2rad(next_psi_dg)
            next_x = position.x + round(self.leg_m*(np.cos(psi_rad)))
            next_y = position.y + round(self.leg_m*(np.sin(psi_rad)))

            for theta in range(-ac_max_theta_dg, ac_max_theta_dg+step_psi, step_psi):
                next_theta_dg = current_theta_dg + theta
                
                if next_theta_dg > 360:
                    next_theta_dg -= 360
                if next_theta_dg < 0:
                    next_theta_dg += 360
                    
                theta_rad = np.deg2rad(next_theta_dg)

                z = round(self.leg_m*(np.cos(theta_rad)))
                
            #     next_x = position.x + x
            #     next_y = position.y + y
                # next_z = position.z + z
                # moves.append([next_x, next_y, next_z])
            
            # next_x = position.x + round(self.leg_m*(np.cos(psi_rad)))
            # next_y = position.y + round(self.leg_m*(np.sin(psi_rad)))
            for z in range(min_z, max_z+step_z, step_z):
                next_z = position.z + z
                moves.append([next_x, next_y, next_z])
            
        return moves
    

class Grid():
    """
    Defines a grid-based environment for navigation by an FWAgent.

    Attributes:
        agent (FWAgent): The agent navigating the grid.
        x_min_m, y_min_m, z_min_m (float): Minimum coordinates of the grid.
        x_max_m, y_max_m, z_max_m (float): Maximum coordinates of the grid.
        offset_x, offset_y, offset_z (float): Offsets for grid coordinates.
        obstacles (list[Obstacle]): List of obstacles within the grid.

    Methods:
        __init__: Constructor to initialize the grid.
        get_grid_size: Returns the size of the grid.
        insert_obstacles: Adds an obstacle to the grid.
        map_position_to_grid: Maps a position to the grid, adjusting based on direction.
        set_grid_size: Sets the size of the grid based on agent constraints.
        is_out_bounds: Checks if a position is out of the grid bounds.
        is_in_obstacle: Checks if a position is within an obstacle.
        convert_position_to_index: Converts a 3D position to a 1D index.
        convert_index_to_position: Converts a 1D index back to a 3D position.
    """

    def __init__(self, 
                 agent:FWAgent, 
                 x_max_m:float=1000, 
                 y_max_m:float=1000,
                 z_max_m:float=1000,
                 x_min_m:float=0,
                 y_min_m:float=0,
                 z_min_m:float=0,
                 offset_x:float=0,
                 offset_y:float=0,
                 offset_z:float=0) -> None:
        
        self.agent = agent
        self.x_min_m = x_min_m
        self.y_min_m = y_min_m
        self.z_min_m = z_min_m

        self.x_max_m = x_max_m
        self.y_max_m = y_max_m
        self.z_max_m = z_max_m
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.offset_z = offset_z

        self.set_grid_size()
        self.obstacles = []

    def get_grid_size(self) -> tuple:
        return (self.sx, self.sy, self.sz)

    def insert_obstacles(self, obstacle:Obstacle) -> None:
        self.obstacles.append(obstacle)

    def map_position_to_grid(self, position:PositionVector, 
                             direction_vector:PositionVector) -> PositionVector:
        """
        sets the position up or down based on set_high
        Note based on unit vector from start to goal 
        
        Check modulus if 0 if so then we are on the grid 
        and snap to grid
        
        if i direction is positive then get ceiling 
        if i direction is negative then get floor

        """
        if position.x % self.sx == 0 and position.y % self.sy == 0 and \
            position.z % self.sz == 0:
            return position
        
        if direction_vector.x > 0:
            x_round = self.sx * m.ceil(position.x/self.sx)
        else:
            x_round = self.sx * m.floor(position.x/self.        sx)
            
        if direction_vector.y > 0:
            y_round = self.sy * m.ceil(position.y/self.sy)
        else:
            y_round = self.sy * m.floor(position.y/self.sy)
        
        if direction_vector.z > 0:
            z_round = self.sz * m.ceil(position.z/self.sz)
        else:
            z_round = self.sz * m.floor(position.z/self.sz)
        
        rounded_position = PositionVector(x_round, y_round, z_round)
        rounded_position.set_position(x_round, y_round, z_round)
        return rounded_position

    def set_grid_size(self) -> None:
        """
        From paper set grid size based on agent constraints
        sx = size of grid in x direction
        sy = size of grid in y direction
        sz = size of grid in z direction
        """
        #round up 
        # self.sx = m.ceil(2/3 * self.agent.horizontal_min_radius_m)
        # self.sy = self.sx
        # self.sz = m.ceil(self.sx * 
        #                 np.tan(np.deg2rad(self.agent.max_climb_angle_dg)))
        self.sx = self.x_max_m - self.x_min_m
        self.sy = self.y_max_m - self.y_min_m
        self.sz = self.z_max_m - self.z_min_m


    def is_out_bounds(self, position:PositionVector) -> bool:
        """
        Check if position is out of bounds 
        """
        if position.x < self.x_min_m or position.x > self.x_max_m:
            return True
        if position.y < self.y_min_m or position.y > self.y_max_m:
            return True
        if position.z < self.z_min_m or position.z > self.z_max_m:
            return True
        
        return False

    def is_in_obstacle(self, position:PositionVector) -> bool:
        """
        Check if position is in obstacle
        """
        for obs in self.obstacles:
            if obs.is_inside2D(position, self.agent.radius_m):
                return True

    def convert_position_to_index(self, position:PositionVector) -> int:
        # """returns 1D index of position"""
        tuple_position = (int(position.x), int(position.y), int(position.z))
        return "_".join(map(str, tuple_position))
        # if position.z == 0:
        #     return int(position.x + position.y * self.sx)
        # else:
        #     index = position.x + (position.y * self.sx) + \
        #         (position.z * self.sx * self.sy)
        #     return int(index)
        
    def convert_index_to_position(self, index:int) -> PositionVector:
        """returns position from 1D index"""

        x = index % self.sx 
        index /= self.sx 
        y = index % self.sy
        z = index / self.sy 

        return PositionVector(int(x), int(y), int(z))
    
        