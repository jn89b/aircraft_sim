from src.guidance_lib.src.Grid import Grid, FWAgent
from src.guidance_lib.src.Radar import Radar
from src.guidance_lib.src.Obstacle import Obstacle
from src.guidance_lib.src.Terrain import Terrain
from src.config.Config import utm_param
from src.guidance_lib.src.PositionVector import PositionVector
from src.guidance_lib.src.SparseAstar import SparseAstar
from src.guidance_lib.src.Config.radar_config import RADAR_AIRCRAFT_HASH_FILE

import pandas as pd
import numpy as np
import time 
import os
import plotly.graph_objects as go
from src.data_vis.DataParser import DataHandler

from concurrent.futures import ThreadPoolExecutor

def run_sparse_astar(radar_heuristic_weight=100):
    # Initialize SparseAstar
    sparse_astar = SparseAstar(grid=grid, terrain_map=grand_canyon, 
                               use_terrain=True, velocity=15,
                               terrain_buffer_m=60, 
                               use_radar=True, radar_info=radars,
                               rcs_hash=rcs_hash,
                               radar_weight=radar_heuristic_weight)

    # Initialize nodes
    sparse_astar.init_nodes()

    # Perform the search
    path = sparse_astar.search()

    #print when done
    print('done')

    # Optionally, you can return the path or handle it here
    return path

#%%

"""
A manual test of the radar system with terrain avoidance using sparse A*
"""
data_handle = DataHandler()
start_position = PositionVector(0, 0, 2000)
goal_position  = PositionVector(1500, 1500, 1580)

# start_position = PositionVector(3750, 300, 1200)
# goal_position  = PositionVector(2000, 4800, 1300)

grand_canyon = Terrain('tif_data/n36_w113_1arc_v3.tif', 
                       lon_min = -112.5, 
                       lon_max = -112.45, 
                       lat_min = 36.2, 
                       lat_max = 36.25,
                       utm_zone=utm_param['grand_canyon'])

# grand_canyon = Terrain('tif_data/n36_w113_1arc_v3.tif', 
#                        lon_min = -112.5, 
#                        lon_max = -112.48, 
#                        lat_min = 36.2, 
#                        lat_max = 36.22,
#                        utm_zone=utm_param['grand_canyon'])

goal_lat, goal_lon = grand_canyon.latlon_from_cartesian(goal_position.x,
                                                        goal_position.y)

goal_elevation  = grand_canyon.get_elevation_from_latlon(goal_lat, goal_lon)

print('goal elevation: ', goal_elevation)

fw_agent_psi_dg = 45
fw_agent = FWAgent(start_position, 0, fw_agent_psi_dg)
fw_agent.vehicle_constraints(horizontal_min_radius_m=60, 
                                max_climb_angle_dg=5,
                                max_psi_turn_dg=25)
fw_agent.leg_m = 25
fw_agent.set_goal_state(goal_position)

## create grid
x_max = int(grand_canyon.max_x - grand_canyon.min_x)
y_max = int(grand_canyon.max_y - grand_canyon.min_y)
z_max = 2100
z_min = 1000

grid = Grid(agent=fw_agent, x_max_m=x_max, y_max_m=y_max, 
            z_max_m=z_max, z_min_m=z_min)

obs_positions = [(1000,1000,10)]
obs_list = []
for pos in obs_positions:
    obs_position = PositionVector(pos[0], pos[1], pos[2])
    radius_obs_m = 5
    some_obstacle = Obstacle(obs_position, radius_obs_m)
    obs_list.append(some_obstacle)
    grid.insert_obstacles(some_obstacle)

radar_pos = PositionVector(500, 500, 1970)
print('radar pos: ', radar_pos.x, radar_pos.y, radar_pos.z)
start_lat, start_lon = grand_canyon.latlon_from_cartesian(radar_pos.x, radar_pos.y) 
start_elevation = grand_canyon.get_elevation_from_latlon(start_lat,start_lon)

print('start elevation: ', start_elevation)

radar_params = {
    'pos': radar_pos,
    'azimuth_angle_dg': 45,
    'elevation_angle_dg': 80, #this is wrt to z axis
    'radar_range_m': 100,    
    'max_fov_dg': 120,  
    'vert_max_fov_dg': 80,
    'c1': -0.29,
    'c2': 1200,
    'radar_fq_hz': 10000,
    'grid': grid
}

radar_params = {
    'pos': radar_pos,
    'azimuth_angle_dg': 45,
    'elevation_angle_dg': 80, #this is wrt to z axis
    'radar_range_m': 100,    
    'max_fov_dg': 120,  
    'vert_max_fov_dg': 80,
    'c1': -0.29,
    'c2': 1200,
    'radar_fq_hz': 10000,
    'grid': grid
}

radar_params = {
    'pos': radar_pos,
    'azimuth_angle_dg': 45,
    'elevation_angle_dg': 80, #this is wrt to z axis
    'radar_range_m': 100,    
    'max_fov_dg': 120,  
    'vert_max_fov_dg': 80,
    'c1': -0.29,
    'c2': 1200,
    'radar_fq_hz': 10000,
    'grid': grid
}

radar1 = Radar(radar_params)
radar2 = Radar(radar_params)

#detection with jit
init_time = time.time()
detection_info_jit = radar1.compute_fov_cells_3d(obs_list, 
                                             True,
                                             True, 
                                             grand_canyon.expanded_array,
                                             np.array([grand_canyon.min_x, grand_canyon.max_x]),
                                             np.array([grand_canyon.min_y, grand_canyon.max_y]))
final_time = time.time() - init_time
print("final time jit: ", final_time)
# detection_info = radar1.compute_fov_cells_3d(obs_list, True,
                   
#%% 
## load hash table for rcs, wrap this in a function
pwd = os.getcwd()
info_dir = 'info/hash/'
save_dir = 'figures/' + RADAR_AIRCRAFT_HASH_FILE
rcs_file = info_dir+ RADAR_AIRCRAFT_HASH_FILE + '.csv'
df = pd.read_csv(rcs_file, header=None)
#get first column
rpy_keys = df.iloc[:, 0]
rcs_vals = df.iloc[:, 1]
max_rcs_val = min(rcs_vals)

#convert to dictionary
rcs_hash = dict(zip(rpy_keys, rcs_vals))

                                    
## load the planner
radars = [radar1, radar2]
# sparse_astar = SparseAstar(grid=grid, terrain_map=grand_canyon, 
#                            use_terrain=True, velocity=15,
#                            terrain_buffer_m=60, 
#                            use_radar=True, radar_info=radars,
#                            rcs_hash=rcs_hash,
#                            radar_weight=100)
# sparse_astar.init_nodes()
# path = sparse_astar.search()

num_worker = 4
weights = [0, 0.1, 0.5, 1]
# Create a ThreadPoolExecutor
paths = []
with ThreadPoolExecutor(max_workers=2) as executor:
    # Start two threads and get the future objects
    # future1 = executor.submit(run_sparse_astar, 50)
    # future2 = executor.submit(run_sparse_astar, 0)

    # # Get the results (paths) from the futures
    # path1 = future1.result()
    # path2 = future2.result()
    
    
    #convert to for looops 
    for i in range(len(weights)):
        paths.append(executor.submit(run_sparse_astar, i*100))
    

scatter_list = []
#make a color list based on the number of paths
color_list = ['red', 'blue', 'green', 'orange', 'purple', 'yellow']
for i, p in enumerate(paths):
    result = data_handle.return_planner_states(p.result())
    planner_state = data_handle.scale_cartesian_with_terrain(result, grand_canyon)
    scatter_list.append(go.Scatter3d(x=planner_state['x'],
                            y=planner_state['y'], 
                            z=planner_state['z'],
                            line=dict(
                                color=color_list[i],
                                width=4
                                ),
                            marker=dict(
                                color=planner_state['z'],
                                colorscale='Viridis',
                                size=3,
                                opacity=0.1,
                                colorbar=dict(
                                    title='Range From Source Radar',
                                    x=0)
                                ),
                            name='planner trajectory'
                            ))
    

# planner_states = data_handle.return_planner_states(path1)
# #save to csv
# planner_states.to_csv('planner_states.csv')
# formatted_states = data_handle.format_traj_data_with_terrain(planner_states, 
#                                                          grand_canyon)


# planner_states2 = data_handle.return_planner_states(path2)
# formatted_states2 = data_handle.format_traj_data_with_terrain(planner_states2,
#                                                             grand_canyon)


# trajectory = go.Scatter3d(x=formatted_states['x'],
#                             y=formatted_states['y'], 
#                             z=planner_states['z'],
#                             line=dict(
#                                 color='red',
#                                 width=4
#                                 ),
#                             marker=dict(
#                                 color=planner_states['z'],
#                                 colorscale='Viridis',
#                                 size=3,
#                                 opacity=0.1,
#                                 colorbar=dict(
#                                     title='Range From Source Radar',
#                                     x=0)
#                                 ),
#                             name='planner trajectory'
#                             )

# trajectory2 = go.Scatter3d(x=formatted_states2['x'],
#                             y=formatted_states2['y'], 
#                             z=planner_states2['z'],
#                             line=dict(
#                                 color='blue',
#                                 width=4
#                                 ),
#                             marker=dict(
#                                 color=planner_states2['z'],
#                                 colorscale='Viridis',
#                                 size=3,
#                                 opacity=0.1,
#                                 colorbar=dict(
#                                     title='Range From Source Radar',
#                                     x=0)
#                                 ),
#                             name='planner trajectory2'
#                             )

                                    
#%% 

detect_info = detection_info_jit.items()
voxels = radar1.get_voxels(detect_info,100)
voxels_normalized = data_handle.format_radar_data_with_terrain(voxels, grand_canyon)
voxel_visualizer = radar1.get_visual_scatter_radar(voxels_normalized)
fig = grand_canyon.plot_3d_expanded(1, 0 , 2200)

#plot the goal position
x_goal = goal_position.x/(grand_canyon.max_x - grand_canyon.min_x) * grand_canyon.expanded_array.shape[0]
y_goal = goal_position.y/(grand_canyon.max_y - grand_canyon.min_y) * grand_canyon.expanded_array.shape[1]
z_goal = goal_position.z

fig.add_trace(go.Scatter3d(x=[x_goal], y=[y_goal], z=[z_goal], mode='markers', marker=dict(color='green', size=5)))


#plot the original position
x_pos = radar_pos.x/(grand_canyon.max_x - grand_canyon.min_x) * grand_canyon.expanded_array.shape[0]
y_pos = radar_pos.y/(grand_canyon.max_y - grand_canyon.min_y) * grand_canyon.expanded_array.shape[1]
z_pos = radar_pos.z

fig.add_trace(go.Scatter3d(x=[x_pos], y=[y_pos], z=[z_pos], mode='markers', marker=dict(color='red', size=5)))
fig.add_trace(voxel_visualizer)
for scatter in scatter_list:
    fig.add_trace(scatter)
    
# fig.add_trace(trajectory)
# fig.add_trace(trajectory2)

#make another plot
regular_voxels = go.Scatter3d(x=voxels['voxel_x'], y=voxels['voxel_y'], z=voxels['voxel_z'],
                              mode='markers', marker=dict(color='blue', size=5))
regular_position = go.Scatter3d(x=[radar_pos.x], y=[radar_pos.y], z=[radar_pos.z], mode='markers', marker=dict(color='red', size=5))
fig2 = go.Figure()
fig2.add_trace(regular_voxels)
fig2.add_trace(regular_position)

fig.show()
# fig2.show()
