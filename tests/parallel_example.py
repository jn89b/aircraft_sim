import numba
from numba_raytrace import is_inside2D_jit, fast_voxel_algo3D_jit, check_obstacles
from numba import jit, prange,njit, typed
import numpy as np
import time as time

def non_parallelized_function(array):
    result = np.zeros_like(array)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            # Your computation goes here
            result[i, j] = array[i, j] * 2  # Replace this with your actual computation
    return result

# Define a function that you want to parallelize
@jit(nopython=True, parallel=True)
def parallelized_function(array):
    result = np.zeros_like(array)
    for i in prange(array.shape[0]):
        for j in prange(array.shape[1]):
            # Your computation goes here
            result[i, j] = array[i, j] * 2  # Replace this with your actual computation
    return result


@njit(fastmath=True)
def compute_radar_orientation(bearing_rad:float, 
                              elevation_rad:float,
                              radar_position:np.ndarray,
                              radar_range:float) -> np.ndarray:
    """
    Returns the radar orientation vector in the local NED frame
    """
    r_max_x = radar_position[0] + radar_range*np.cos(bearing_rad) * \
        np.sin(elevation_rad)
        
    r_max_y = radar_position[1] + radar_range*np.sin(bearing_rad) * \
        np.sin(elevation_rad)
        
    r_max_z = radar_position[2] + radar_range*np.cos(elevation_rad)
    
    #round to nearest whole number
    r_max_x = np.round(r_max_x)
    r_max_y = np.round(r_max_y)
    r_max_z = np.round(r_max_z)
    
    return np.array([r_max_x, r_max_y, r_max_z])
    

@njit(parallel=True)
def get_radar_fov_vals(azmith_bearing_dgs:np.ndarray,
                       elevation_bearing_dgs:np.ndarray,
                       radar_position:np.ndarray,
                       radar_range:float, 
                       obstacle_list:np.ndarray) -> np.ndarray:
    
    """
    Loop through bearing then loop through elevation

    """
    
    values = np.arange(1,9)
    result = np.zeros((len(azmith_bearing_dgs), len(elevation_bearing_dgs),
                       radar_range))

    for i in prange(len(azmith_bearing_dgs)):
        for j in prange(len(elevation_bearing_dgs)):
            bearing_rad = np.deg2rad(azmith_bearing_dgs[i])
            elevation_rad = np.deg2rad(elevation_bearing_dgs[j])
            
            radar_bounds = compute_radar_orientation(bearing_rad, 
                                                     elevation_rad, 
                                                     radar_position, 
                                                     radar_range)
            
            x_0 = radar_position[0]
            y_0 = radar_position[1]
            z_0 = radar_position[2]
            
            x_1 = radar_bounds[0]
            y_1 = radar_bounds[1]
            z_1 = radar_bounds[2]
            
            #store the cell rays
            cell_rays = fast_voxel_algo3D_jit(x_0, y_0, z_0, x_1, y_1, z_1, 
                                              obstacle_list)
            
            print("cell rays", cell_rays.shape) 
            result[i,j,:] = cell_rays
            
    return result


# radar_range = 100
# bearing_rad = np.deg2rad(45)
# elevation_rad = np.deg2rad(45)
# radar_position = np.array([0,0,0])
#radar_bounds = compute_radar_orientation(bearing_rad, elevation_rad, radar_position, radar_range)


obstacles = []
N_obstacles = 50
for i in range(N_obstacles):
    x = np.random.randint(50, 100)
    y = np.random.randint(50, 100)
    z = np.random.randint(50, 100)
    # if i == 0:
    #     x = 10
    #     y = 10
    #     z = 10
        
    radius = 2
    obstacles.append((x,y,z, radius))
    
typed_obstacle_list = typed.List(obstacles)

min_azmith_dg = 0
max_azmith_dg = 10
min_elevation_dg = 0
max_elevation_dg = 10
radar_position = np.array([0,0,0])
radar_range = 100

azmith_ranges = np.arange(min_azmith_dg, max_azmith_dg)
elevation_ranges = np.arange(min_elevation_dg, max_elevation_dg)

results = get_radar_fov_vals(azmith_ranges, elevation_ranges, 
                             radar_position, radar_range, typed_obstacle_list)

min_azmith_dg = 0
max_azmith_dg = 90
min_elevation_dg = 0
max_elevation_dg = 90
radar_position = np.array([0,0,0])
radar_range = 100

#%%
start_time = time.time()
azmith_ranges = np.arange(min_azmith_dg, max_azmith_dg)
elevation_ranges = np.arange(min_elevation_dg, max_elevation_dg)

results = get_radar_fov_vals(azmith_ranges, elevation_ranges, 
                             radar_position, radar_range, typed_obstacle_list)

end_time = time.time() - start_time
print("Time for bearing and azmith:", end_time)

# Create a sample 2D array
data = np.random.rand(2, 2)

# Call the parallelized function
# result = parallelized_function(data)

new_data = np.random.rand(7500, 7500)

# start_time = time.time()
# non_parallelized_function(new_data)
# print("Non parallelized time:", time.time() - start_time)

# start_time = time.time()
# parallelized_function(new_data)
# print("Parallelized time:", time.time() - start_time)


