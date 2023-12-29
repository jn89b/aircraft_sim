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

    x = int(m.floor(x0))
    y = int(m.floor(y0))
    z = int(m.floor(z0))

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
    for i in range(n):        
        curr_pos = np.array([x,y,z])

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
    """
    
    returns a numpy array of the cell rays
    where each row is a cell ray and the columns are the x,y,z coordinates
    that is column 0 is x, column 1 is y, column 2 is z
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)

    x = int(m.floor(x0))
    y = int(m.floor(y0))
    z = int(m.floor(z0))

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
        n += int(m.floor(x1)) - x
        t_next_horizontal = (m.floor(x0) + 1 - x0) * dt_dx
    else:
        x_inc = -1
        n += x - int(m.floor(x1))
        t_next_horizontal = (x0 - m.floor(x0)) * dt_dx

    if (dy == 0):
        y_inc = 0
        t_next_vertical = dt_dy 
    elif (y1 > y0):
        y_inc = 1
        n += int(m.floor(y1)) - y
        t_next_vertical = (m.floor(y0) + 1 - y0) * dt_dy
    else:
        y_inc = -1
        n += y - int(m.floor(y1))
        t_next_vertical = (y0 - m.floor(y0)) * dt_dy

    if (dz == 0):
        z_inc = 0
        t_next_height = dt_dz 
    elif (z1 > z0):
        z_inc = 1
        n += int(m.floor(z1)) - z
        t_next_height = (m.floor(z0) + 1 - z0) * dt_dz
    else:
        z_inc = -1
        n += z - int(m.floor(z1))
        t_next_height = (z0 - m.floor(z0)) * dt_dz


    # cell_rays = []
    #empty numpy array
    #check if current position is 10 10 10
    cell_rays = np.empty((n,3))
    for i in range(n):        
        current_position = np.array([x,y,z])
        if obs_list:
            for obs in obs_list:
                obs_position = np.array([obs[0], obs[1], obs[2]])
                obs_radius = obs[3]
                if is_inside2D_jit(current_position, obs_position, obs_radius) == True:
                    return cell_rays[:i]
                    
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
        
@njit
def dumb_function(yes:bool) -> bool:
    if yes == True:
        return True
    else:
        return False

#%% Checking obstacle inside with no jit and jit
obstacles = []
N_obstacles = 20
for i in range(N_obstacles):
    x = np.random.randint(500, 1000)
    y = np.random.randint(500, 1000)
    z = np.random.randint(500, 1000)

    radius = 2
    obstacles.append((x,y,z, radius))
    
obstacles.append((615,615,615, radius))

typed_obstacle_list = typed.List(obstacles)
current_position = np.array([10,15,30])

dumb_function(True)

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

x_end = 1000
y_end = 1000
z_end = 1000

PLOT = True

start_time = time.time()
cell_rays_no_jit = fast_voxel_algo3D(x_0, y_0, z_0, x_end, y_end, z_end, obstacles)
no_jit = time.time() - start_time
print("--- %s seconds with no jit ---" % (time.time() - start_time))

#precompile with jit
x_end = 10
y_end = 10
z_end = 10
cell_rays_jit = fast_voxel_algo3D_jit(x_0, y_0, z_0, x_end, y_end, z_end, typed_obstacle_list)   

x_end = 1000
y_end = 1000
z_end = 1000
start_time = time.time()
cell_rays_jit = fast_voxel_algo3D_jit(x_0, y_0, z_0, x_end, y_end, z_end, typed_obstacle_list)   
jit_time = time.time() - start_time
#remove nans from cell rays

print("--- %s seconds with jit ---" % jit_time)
print("speedup raytrace:", no_jit/jit_time)

array_jit = cell_rays_jit


#%% Plotting
#plot the results with plotly
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

#convert to pandas dataframe
cell_rays_jit = pd.DataFrame(cell_rays_jit, columns=['x','y','z'])
if PLOT == True:
    fig = px.scatter_3d(cell_rays_jit, x='x', y='y', z='z')
    
    #plot obstacles
    obs_df = pd.DataFrame(obstacles, columns=['x','y','z','radius'])
    
    fig.add_trace(go.Scatter3d(x=obs_df['x'], y=obs_df['y'], z=obs_df['z'],
                                 mode='markers', marker=dict(size=obs_df['radius']*10)))
    
    fig.show()
 
    
#%% Regular radar method vs numba raytrace
from numba import cuda

azmith_bearing_dgs = np.arange(0, 15)
elevation_bearing_dgs = np.arange(0,10)
radar_range = 20

size_radar = (len(azmith_bearing_dgs)) * (len(elevation_bearing_dgs)) #* radar_range
n_length = 2500
position_size = 3

#this will be a 3d array
#first dimension is the azmith
#second dimension is the elevation
#third dimension is the range 

radar_fov_cells = np.zeros((n_length, 
                            position_size, 
                            size_radar))


jit_list = []
init_time = time.time()
for i in range(len(azmith_bearing_dgs)):
    for j in range(len(elevation_bearing_dgs)):
        bearing_rays = fast_voxel_algo3D_jit(x_0, y_0, z_0,
                                            x_end, y_end, z_end,
                                            typed_obstacle_list)
        
        jit_list.append(bearing_rays)
final_time = time.time() - init_time
print("jit time:", final_time)

#%% parallelize the for loop with jit
@njit(parallel=True)


#%%
@cuda.jit
def add_kernel(radar_device:np.ndarray, typed_obstacle_list:np.ndarray,
               x0:float, y0:float, z0:float, x_end:float, y_end:float, z_end:float):
    """
    from the index get the azmith and elevation
    feed that into the fast_voxel_algo3D_jit
    https://numba.readthedocs.io/en/stable/cuda/examples.html#jit-function-cpu-gpu-compatibility
    """
    i_start = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    threads_per_grid = cuda.blockDim.x * cuda.gridDim.x
    for i in range(i_start, radar_device.shape[0], threads_per_grid):
        angles = index_to_angle(i)
        bearing_rays = fast_voxel_algo3D_jit(x_0, y_0, z_0,
                                            x_end, y_end, z_end,
                                            typed_obstacle_list)

        radar_device[:,:,i] = 2


@njit
def angle_to_index(azmith_index:int, elevation_index:int) -> int:
    """Converts the azmith and elevation index to the index in the 3d array

    Args:
        azmith_index (int): azmith index
        elevation_index (int): elevation index

    Returns:
        int: index in the 3d array
    """
    return azmith_index * len(elevation_bearing_dgs) + elevation_index

@njit
def index_to_angle(index:int) -> tuple:
    """Converts the index in the 3d array to the azmith and elevation index

    Args:
        index (int): index in the 3d array

    Returns:
        tuple: (azmith_index, elevation_index)
    """
    azmith_index = index // len(elevation_bearing_dgs)
    elevation_index = index % len(elevation_bearing_dgs)
    return (azmith_index, elevation_index)


some_array = []
for i in range(len(azmith_bearing_dgs)):
    for j in range(len(elevation_bearing_dgs)):
        idx = angle_to_index(i,j)
        angles = index_to_angle(idx)
        
        print("azmith:", angles[0], "elevation:", angles[1])
        print("idx:", idx)
        total = angles[0] + angles[1]
        some_array.append(total)
    
# azmith_device = cuda.to_device(azmith_bearing_dgs)
# elevation_device = cuda.to_device(elevation_bearing_dgs)

gpu = cuda.get_current_device()

radar_fov_device  = cuda.to_device(radar_fov_cells)
test_array = np.zeros(size_radar, dtype=np.int8)
test_array_device = cuda.to_device(test_array)


block_size = 256
num_blocks = (n_length + (block_size - 1)) // block_size
num_blocks = 32 * 80  #

add_kernel[num_blocks, block_size](radar_fov_device, typed_obstacle_list,
                                   x_0, y_0, z_0, x_end, y_end, z_end)

radar_fov_cells = radar_fov_device.copy_to_host()

#test_array = test_array_device.copy_to_host()


# #check if test_array is the same as some_array
# for i in range(len(test_array)):
#     if test_array[i] != some_array[i]:
#         print("not the same")
#         break

        

# not_jit_list = []
# init_time = time.time()
# for i in range(len(azmith_bearing_dgs)):
#     for j in range(len(elevation_bearing_dgs)):
#         bearing_rays = fast_voxel_algo3D(x_0, y_0, z_0, x_end, y_end, z_end, obstacles)

        
#         # radar_fov_cells[i,j,:] = bearing_rays
#         not_jit_list.append(bearing_rays)
# final_time = time.time() - init_time
# print("jit time:", final_time)

