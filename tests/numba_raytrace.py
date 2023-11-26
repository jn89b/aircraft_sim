from numba import jit, njit, vectorize
from numba import typed, types
import numpy as np
import time 
import sys
import math as m
# from src.guidance_lib.src.PositionVector import PositionVector
def fast_voxel_algo3D(x0:float, y0:float, z0:float, 
                      x1:float, y1:float, z1:float, 
                      obs_list=[]) ->list:
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)

    x = int(np.floor(x0))
    y = int(np.floor(y0))
    z = int(np.floor(z0))

    if dx == 0:
        dt_dx = 1000000
    else:
        dt_dx = 1/ dx
    
    if dy == 0:
        dt_dy = 1000000
    else:
        dt_dy = 1/ dy
    
    if dz == 0:
        dt_dz = 1000000
    else:
        dt_dz = 1/ dz

    t = 0
    n = 1 
    t_next_horizontal = 0
    t_next_vertical = 0
    t_next_height = 0

    if (dx == 0):
        x_inc = 0
        t_next_horizontal = dt_dx 
    elif (x1 > x0):
        x_inc = 1
        n += int(np.floor(x1)) - x
        t_next_horizontal = (np.floor(x0) + 1 - x0) * dt_dx
    else:
        x_inc = -1
        n += x - int(np.floor(x1))
        t_next_horizontal = (x0 - np.floor(x0)) * dt_dx

    if (dy == 0):
        y_inc = 0
        t_next_vertical = dt_dy 
    elif (y1 > y0):
        y_inc = 1
        n += int(np.floor(y1)) - y
        t_next_vertical = (np.floor(y0) + 1 - y0) * dt_dy
    else:
        y_inc = -1
        n += y - int(np.floor(y1))
        t_next_vertical = (y0 - np.floor(y0)) * dt_dy

    if (dz == 0):
        z_inc = 0
        t_next_height = dt_dz 
    elif (z1 > z0):
        z_inc = 1
        n += int(np.floor(z1)) - z
        t_next_height = (np.floor(z0) + 1 - z0) * dt_dz
    else:
        z_inc = -1
        n += z - int(np.floor(z1))
        t_next_height = (z0 - np.floor(z0)) * dt_dz


    cell_rays = []
    curr_pos = np.array([x,y,z])
    for i in range(n):        
        if obs_list:
            for obs in obs_list:
                obs_position = np.array([obs[0], obs[1], obs[2]])
                obs_radius = obs[3]
                if is_inside2D(curr_pos, obs_position, obs_radius) == True:
                    return cell_rays

        cell_rays.append((x,y,z))
        # dy < dx then I need to go up since 
        if (t_next_horizontal < t_next_vertical):
            if (t_next_horizontal < t_next_height):
                x += x_inc
                t = t_next_horizontal
                t_next_horizontal += dt_dx
            else:
                z += z_inc
                t = t_next_height
                t_next_height += dt_dz
        else:
            if (t_next_vertical < t_next_height):
                y += y_inc
                t = t_next_vertical
                t_next_vertical += dt_dz
            else:
                z += z_inc
                t = t_next_height
                t_next_height += dt_dz
            
        # print(n)
    return cell_rays

@njit
def fast_voxel_algo3D_jit(x0:float, y0:float, z0:float, 
                      x1:float, y1:float, z1:float, 
                      obs_list:np.ndarray=np.array([])) ->np.ndarray:
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)

    x = int(np.floor(x0))
    y = int(np.floor(y0))
    z = int(np.floor(z0))

    if dx == 0:
        dt_dx = 1000000
    else:
        dt_dx = 1/ dx
    
    if dy == 0:
        dt_dy = 1000000
    else:
        dt_dy = 1/ dy
    
    if dz == 0:
        dt_dz = 1000000
    else:
        dt_dz = 1/ dz

    t = 0
    n = 1 
    t_next_horizontal = 0
    t_next_vertical = 0
    t_next_height = 0

    if (dx == 0):
        x_inc = 0
        t_next_horizontal = dt_dx 
    elif (x1 > x0):
        x_inc = 1
        n += int(np.floor(x1)) - x
        t_next_horizontal = (np.floor(x0) + 1 - x0) * dt_dx
    else:
        x_inc = -1
        n += x - int(np.floor(x1))
        t_next_horizontal = (x0 - np.floor(x0)) * dt_dx

    if (dy == 0):
        y_inc = 0
        t_next_vertical = dt_dy 
    elif (y1 > y0):
        y_inc = 1
        n += int(np.floor(y1)) - y
        t_next_vertical = (np.floor(y0) + 1 - y0) * dt_dy
    else:
        y_inc = -1
        n += y - int(np.floor(y1))
        t_next_vertical = (y0 - np.floor(y0)) * dt_dy

    if (dz == 0):
        z_inc = 0
        t_next_height = dt_dz 
    elif (z1 > z0):
        z_inc = 1
        n += int(np.floor(z1)) - z
        t_next_height = (np.floor(z0) + 1 - z0) * dt_dz
    else:
        z_inc = -1
        n += z - int(np.floor(z1))
        t_next_height = (z0 - np.floor(z0)) * dt_dz


    # cell_rays = []
    #empty numpy array
    current_position = np.array([x,y,z])
    #check if current position is 10 10 10
    cell_rays = np.empty((n,3))
    for i in range(n):        
        if obs_list:
            for obs in obs_list:
                obs_position = np.array([obs[0], obs[1], obs[2]])
                obs_radius = obs[3]
                if is_inside2D_jit(current_position, obs_position, obs_radius) == True:
                    return cell_rays
                    
        # cell_rays.append((x,y,z))
        cell_rays[i,0] = x
        cell_rays[i,1] = y
        cell_rays[i,2] = z
        
        # dy < dx then I need to go up since 
        if (t_next_horizontal < t_next_vertical):
            if (t_next_horizontal < t_next_height):
                x += x_inc
                t = t_next_horizontal
                t_next_horizontal += dt_dx
            else:
                z += z_inc
                t = t_next_height
                t_next_height += dt_dz
        else:
            if (t_next_vertical < t_next_height):
                y += y_inc
                t = t_next_vertical
                t_next_vertical += dt_dz
            else:
                z += z_inc
                t = t_next_height
                t_next_height += dt_dz
            
        # print(n)
    return cell_rays


def is_inside2D(current_position:np.ndarray, 
                obstacle_position:np.ndarray, 
                radius_obs_m:float) ->bool:
    """Check if current position is inside obstacle

    Args:
        current_position (np.ndarray): current position
        obstacle_position (np.ndarray): obstacle position
        radius_obs_m (float): obstacle radius

    Returns:
        bool: True if inside, False if outside
    """
    #check if inside obstacle
    if np.linalg.norm(current_position - obstacle_position) < radius_obs_m:
        return True
    else:
        return False
    

@njit 
def is_inside2D_jit(current_position:np.ndarray, 
                obstacle_position:np.ndarray, 
                radius_obs_m:float) ->bool:
    
    dx = current_position[0] - obstacle_position[0]
    dy = current_position[1] - obstacle_position[1]
    dist = m.sqrt(dx**2 + dy**2)
    if dist <= radius_obs_m:
        # print("dx", dx)
        # print("dy", dy)
        # print("current_position", current_position)
        # print("obstacle_position", obstacle_position)
        return True
    else:
        return False
    
@njit
def check_obstacles(obstacle_list:np.ndarray,
                    obstacle_radius:float,
                    current_position:np.ndarray) ->bool:
    
    for i in range(len(obstacle_list)):
        if is_inside2D_jit(current_position, obstacle_list[i], obstacle_radius):
            return True
        
    return False
        



#%% Checking obstacle inside with no jit and jit
obstacles = []
N_obstacles = 50
for i in range(N_obstacles):
    x = np.random.randint(500, 1000)
    y = np.random.randint(500, 1000)
    z = np.random.randint(500, 1000)
    if i == 0:
        x = 10
        y = 10
        z = 10
        
    radius = 2
    obstacles.append((x,y,z, radius))
    
typed_obstacle_list = typed.List(obstacles)
current_position = np.array([10,15,30])

# #no jit
# start_time = time.time()
# for obs in obstacles:
#     obstacle_position = np.array([obs[0], obs[1], obs[2]])
#     radius_obs_m = 5
#     is_inside2D(current_position, obstacle_position, radius_obs_m)
# no_jit = time.time() - start_time
# print("--- %s Obstacle seconds with no jit ---" % (time.time() - start_time))

# is_inside2D_jit(current_position, obstacle_position, 
#                 radius_obs_m)
# check_obstacles(np.array(obstacles), 5.0, current_position)

# #jit
# start_time = time.time()
# check_obstacles(np.array(obstacles), 5.0, current_position)
# end_time = time.time() - start_time
# print("--- %s Obstacle seconds wit  jit ---" % (end_time))
# print("Obstacle speedup:", no_jit/end_time)

#%% Raytrace with no jit and jit
x_0 = 0
y_0 = 0
z_0 = 0

x_end = 250
y_end = 250
z_end = 250

PLOT = False

start_time = time.time()
cell_rays_no_jit = fast_voxel_algo3D(x_0, y_0, z_0, x_end, y_end, z_end, obstacles)
no_jit = time.time() - start_time
print("--- %s seconds with no jit ---" % (time.time() - start_time))

#precompile with jit
x_end = 10
y_end = 10
z_end = 10
cell_rays_jit = fast_voxel_algo3D_jit(x_0, y_0, z_0, x_end, y_end, z_end, typed_obstacle_list)   

x_end = 250
y_end = 250
z_end = 250
start_time = time.time()
cell_rays_jit = fast_voxel_algo3D_jit(x_0, y_0, z_0, x_end, y_end, z_end, typed_obstacle_list)   
jit_time = time.time() - start_time
print("--- %s seconds with jit ---" % jit_time)
print("speedup raytrace:", no_jit/jit_time)

#%% Plotting
#plot the results with plotly
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

#convert to pandas dataframe
cell_rays_jit = pd.DataFrame(cell_rays_jit, columns=['x','y','z'])
if PLOT == True:
    fig = px.scatter_3d(cell_rays_jit, x='x', y='y', z='z')
    fig.show()