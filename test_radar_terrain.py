from src.guidance_lib.src.Grid import Grid, FWAgent
from src.guidance_lib.src.Radar import Radar
from src.guidance_lib.src.Obstacle import Obstacle
from src.guidance_lib.src.Terrain import Terrain
from src.config.Config import utm_param
from src.guidance_lib.src.PositionVector import PositionVector

import numpy as np
import time 

import plotly.graph_objects as go
#%%

"""
A manual test to see if radar will detect obstacles and terrain
"""
start_position = PositionVector(0, 0, 2000)
goal_position  = PositionVector(3500, 3500, 1200)

# start_position = PositionVector(3750, 300, 1200)
# goal_position  = PositionVector(2000, 4800, 1300)



grand_canyon = Terrain('tif_data/n36_w113_1arc_v3.tif', 
                       lon_min = -112.5, 
                       lon_max = -112.45, 
                       lat_min = 36.2, 
                       lat_max = 36.25,
                       utm_zone=utm_param['grand_canyon'])


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

radar_pos = PositionVector(500, 500, 1950)
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
radar1 = Radar(radar_params)
# obs_list = []

#detection with jit
init_time = time.time()
detection_info_jit = radar1.compute_fov_cells_3d(obs_list, True,
                                             True, grand_canyon.expanded_array,
                                             np.array([grand_canyon.min_x, grand_canyon.max_x]),
                                             np.array([grand_canyon.min_y, grand_canyon.max_y]))
final_time = time.time() - init_time
print("final time jit: ", final_time)
# detection_info = radar1.compute_fov_cells_3d(obs_list, True,
                                    
#%% 
from src.data_vis.DataParser import DataHandler
data_handle = DataHandler()
detect_info = detection_info_jit.items()
voxels = radar1.get_voxels(detect_info,100)
voxels_normalized = data_handle.format_radar_data_with_terrain(voxels, grand_canyon)
voxel_visualizer = radar1.get_visual_scatter_radar(voxels_normalized)
fig = grand_canyon.plot_3d_expanded(1, 0 , 2200)

#plot the original position
x_pos = radar_pos.x/(grand_canyon.max_x - grand_canyon.min_x) * grand_canyon.expanded_array.shape[0]
y_pos = radar_pos.y/(grand_canyon.max_y - grand_canyon.min_y) * grand_canyon.expanded_array.shape[1]
z_pos = radar_pos.z

fig.add_trace(go.Scatter3d(x=[x_pos], y=[y_pos], z=[z_pos], mode='markers', marker=dict(color='red', size=5)))
fig.add_trace(voxel_visualizer)

#make another plot
regular_voxels = go.Scatter3d(x=voxels['voxel_x'], y=voxels['voxel_y'], z=voxels['voxel_z'],
                              mode='markers', marker=dict(color='blue', size=5))
regular_position = go.Scatter3d(x=[radar_pos.x], y=[radar_pos.y], z=[radar_pos.z], mode='markers', marker=dict(color='red', size=5))
fig2 = go.Figure()
fig2.add_trace(regular_voxels)
fig2.add_trace(regular_position)

fig.show()
fig2.show()
