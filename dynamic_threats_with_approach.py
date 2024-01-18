from src.guidance_lib.src.Grid import Grid, FWAgent
from src.guidance_lib.src.Radar import Radar
from src.guidance_lib.src.Obstacle import Obstacle
from src.guidance_lib.src.Terrain import Terrain
from src.config.Config import utm_param
from src.guidance_lib.src.PositionVector import PositionVector
from src.guidance_lib.src.SparseAstar import SparseAstar
from src.guidance_lib.src.Config.radar_config import RADAR_AIRCRAFT_HASH_FILE
from src.guidance_lib.src.SentryThreats import SentryThreats
from src.guidance_lib.src.utils import create_cylinder
from src.guidance_lib.src.OptimalApproach import OptimalApproach

import multiprocessing
import pandas as pd
import numpy as np
import time 
import os
import plotly.graph_objects as go
from src.data_vis.DataParser import DataHandler

from concurrent.futures import ThreadPoolExecutor
#interpolate the data
from scipy.interpolate import interp1d

def run_different_goals(goalPostiion:PositionVector):
    # Initialize SparseAstar
    # sparse_astar = SparseAstar(grid=grid, terrain_map=grand_canyon, 
    #                            use_terrain=True, velocity=15,
    #                            terrain_buffer_m=60, 
    #                            use_radar=True, radar_info=radars,
    #                            rcs_hash=rcs_hash,
    #                            radar_weight=100)
    radar_heuristic_weight = 10    
    sparse_astar = SparseAstar(grid=grid, terrain_map=grand_canyon, 
                                use_terrain=True, velocity=15,
                                terrain_buffer_m=60, 
                                use_radar=True, radar_info=radars,
                                rcs_hash=rcs_hash,
                                radar_weight=radar_heuristic_weight,
                                dynamic_threats=threats,
                                dynamic_threat_weight=10,
                                use_dynamic_threats=True)

    # Initialize nodes
    sparse_astar.init_nodes()

    # Perform the search
    path = sparse_astar.search(use_diff_goal=True, 
                               goal_position=goalPostiion)

    #print when done
    print('done')

    # Optionally, you can return the path or handle it here
    return path

def run_different_dynamic_threats(threats:list, dynamic_threat_weight:int, 
                                  goalPosition:PositionVector):
    # Initialize SparseAstar
    # sparse_astar = SparseAstar(grid=grid, terrain_map=grand_canyon, 
    #                            use_terrain=True, velocity=15,
    #                            terrain_buffer_m=60, 
    #                            use_radar=True, radar_info=radars,
    #                            rcs_hash=rcs_hash,
    #                            radar_weight=100)
    radar_heuristic_weight = 10    
    sparse_astar = SparseAstar(grid=grid, terrain_map=grand_canyon, 
                                use_terrain=True, velocity=15,
                                terrain_buffer_m=60, 
                                use_radar=True, radar_info=radars,
                                rcs_hash=rcs_hash,
                                radar_weight=radar_heuristic_weight,
                                dynamic_threats=threats,
                                dynamic_threat_weight=dynamic_threat_weight,
                                use_dynamic_threats=True)

    # Initialize nodes
    sparse_astar.init_nodes()

    # # Perform the search
    # path = sparse_astar.search()
    # Perform the search
    path = sparse_astar.search(use_diff_goal=True, 
                               goal_position=goalPosition)

    #print when done
    print('done')

    # Optionally, you can return the path or handle it here
    return path
    

def run_sparse_astar(radar_heuristic_weight=100):
    # Initialize SparseAstar
    sparse_astar = SparseAstar(grid=grid, terrain_map=grand_canyon, 
                               use_terrain=True, velocity=35,
                               terrain_buffer_m=60, 
                               use_radar=True, radar_info=radars,
                               rcs_hash=rcs_hash,
                               radar_weight=radar_heuristic_weight,
                               dynamic_threats=threats,
                               dynamic_threat_weight=10,
                               use_dynamic_threats=True)

    # Initialize nodes
    sparse_astar.init_nodes()

    # Perform the search
    path = sparse_astar.search()

    #print when done
    print('done')

    # Optionally, you can return the path or handle it heres
    return path

#%%
n_cpus = multiprocessing.cpu_count()
n_cpus = n_cpus - 5
if n_cpus < 1:
    n_cpus = 1

n_cpus = 4
data_handle = DataHandler()
start_position = PositionVector(0, 0, 2000)
goal_position  = PositionVector(1500, 1500, 1580)
grand_canyon = Terrain('tif_data/n36_w113_1arc_v3.tif', 
                       lon_min = -112.5, 
                       lon_max = -112.48, 
                       lat_min = 36.2, 
                       lat_max = 36.22,
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

obs_positions = [(1250,1250,100)]
obs_list = []
for pos in obs_positions:
    obs_position = PositionVector(pos[0], pos[1], pos[2])
    radius_obs_m = 50
    some_obstacle = Obstacle(obs_position, radius_obs_m)
    obs_list.append(some_obstacle)
    grid.insert_obstacles(some_obstacle)

radar_pos = PositionVector(500, 500, 1970)
radar_pos2 = PositionVector(800, 750, 1950)
print('radar pos: ', radar_pos2.x, radar_pos2.y, radar_pos2.z)
start_lat, start_lon = grand_canyon.latlon_from_cartesian(radar_pos2.x, radar_pos2.y) 
start_elevation = grand_canyon.get_elevation_from_latlon(start_lat,start_lon)

print('start elevation: ', start_elevation)

radar_params = {
    'pos': radar_pos,
    'azimuth_angle_dg': 45,
    'elevation_angle_dg': 80, #this is wrt to z axis
    'radar_range_m': 200,    
    'max_fov_dg': 120,  
    'vert_max_fov_dg': 80,
    'c1': -0.29,
    'c2': 1200,
    'radar_fq_hz': 10000,
    'grid': grid
}

radar_params2 = {
    'pos': radar_pos2,
    'azimuth_angle_dg': 45,
    'elevation_angle_dg': 80, #this is wrt to z axis
    'radar_range_m': 200,    
    'max_fov_dg': 120,  
    'vert_max_fov_dg': 80,
    'c1': -0.29,
    'c2': 1200,
    'radar_fq_hz': 10000,
    'grid': grid
}

radar1 = Radar(radar_params)
radar2 = Radar(radar_params2)
# obs_list = []

#detection with jit
init_time = time.time()
detection_info_jit = radar1.compute_fov_cells_3d(obs_list, True,
                                             True, grand_canyon.expanded_array,
                                             np.array([grand_canyon.min_x, grand_canyon.max_x]),
                                             np.array([grand_canyon.min_y, grand_canyon.max_y]))

detection_info_jit2 = radar2.compute_fov_cells_3d(obs_list, True,
                                                True, grand_canyon.expanded_array,
                                                np.array([grand_canyon.min_x, grand_canyon.max_x]),
                                                np.array([grand_canyon.min_y, grand_canyon.max_y]))
final_time = time.time() - init_time
print("final time jit: ", final_time)
# detection_info = radar1.compute_fov_cells_3d(obs_list, True,

## DYNAMIC THREATS
threat_1_position = PositionVector(300, 450, 2000)
threat_1_velocity = 0.002
threat_1_heading = 0
use_ellipse = True
ellipse_params = {'a': 300, 'b': 100}
threat_1_radar_params = {
    'pos': threat_1_position,
    'azimuth_angle_dg': 0,
    'elevation_angle_dg': 0, #this is wrt to z axis
    'radar_range_m': 250,    
    'max_fov_dg': 120,  
    'vert_max_fov_dg': 80,
    'c1': -0.29,
    'c2': 1200,
    'radar_fq_hz': 10000,
    'grid': None
}

ellipse_params_2 = {'a': 200, 'b': 200}

threat_2_position = PositionVector(800, 800, 1975)
threat_1 = SentryThreats(threat_1_position.vec, threat_1_position.vec, threat_1_velocity,
                        threat_1_heading, use_ellipse, ellipse_params, threat_1_radar_params)



threat_2_velocity = 0.00099
threat_2_radar_params = {
    'pos': threat_1_position,
    'azimuth_angle_dg': 0,
    'elevation_angle_dg': 0, #this is wrt to z axis
    'radar_range_m': 150,    
    'max_fov_dg': 120,  
    'vert_max_fov_dg': 80,
    'c1': -0.29,
    'c2': 1200,
    'radar_fq_hz': 10000,
    'grid': None
}
threat_2 = SentryThreats(threat_2_position.vec, threat_1_position.vec, threat_2_velocity,
                        threat_1_heading, use_ellipse, ellipse_params_2, threat_2_radar_params,
                        reverse=True)


# threat_2 = SentryThreats(threat_2_position.vec, threat_1_position.vec, threat_1_velocity,
#                         threat_1_heading, use_ellipse, ellipse_params_2, threat_1_radar_params)

threats = [threat_1, threat_2]


# find optimal approaches
# find optimal approaches
approach_params = {
    'pos': goal_position,
    'azimuth_angle_dg': 45,
    'elevation_angle_dg': 90,
    'max_effect_range_m': 100,
    'max_fov_dg': 359,
    'vert_max_fov_dg': 60,
    'grid': None,
    'effector_power': 50,
    'fov_dg_steps': 15,
    'vert_fov_dg_steps': 15
}

required_sum_power_density = 1.0

ag = OptimalApproach(approach_params, use_terrain=True, terrain_map=grand_canyon)
detection_info,detection_priority = ag.get_possible_approaches(
    required_sum_power_density, obs_list)
best_approaches = ag.find_optimal_approach_vectors(n_cpus)
approach_goal_list = []   
for approach in best_approaches:
    print('approach: ', approach[0])
    approach_position = PositionVector(approach[0][0], approach[0][1], approach[0][2])
    approach_goal_list.append(approach_position)
    
                
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

#weights = [0, 0.1, 0.5, 1]
dynamic_threat_weights = [1000]

goal_1 = PositionVector(1500, 1500, 1580)
goal_2 = PositionVector(2500, 2500, 1600)
goal_list = [goal_1, goal_2]
# Create a ThreadPoolExecutor
paths = []
init_time = time.time()
with ThreadPoolExecutor(max_workers=4) as executor:
    # Start two threads and get the future objects
    # for i in range(len(goal_list)):
    #     paths.append(executor.submit(run_different_goals, goal_list[i]))

    for i in range(len(approach_goal_list)):
        process = executor.submit(run_different_dynamic_threats, threats, 
                                  dynamic_threat_weights[0], approach_goal_list[i])
        paths.append(process)
    # for i in range(len(dynamic_threat_weights)):
    #     paths.append(executor.submit(run_different_dynamic_threats, threats, 
    #                                  dynamic_threat_weights[i]))

colorscale = 'HSV'
#%% 
final_time = time.time() - init_time
print("final time to generate paths: ", final_time)
scatter_list = []
time_list = []
#make a color list based on the number of paths
color_list = ['blue', 'green', 'orange', 'purple', 'yellow']
for i, p in enumerate(paths):
    results = p.result()
    traj_time = results[:,9]
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
                                colorscale=colorscale,
                                size=3,
                                opacity=0.1,
                                ),
                            name='planner trajectory'
                            ))
    time_list.append(traj_time)

#%% 
import matplotlib.pyplot as plt

result = p.result()
x = result[:,0]
y = result[:,1]
z = result[:,2]
phi = result[:,3]
theta = result[:,4]
psi = result[:,5]
time_vector = result[:,9]

#interpolate the trajectory
time_interp = np.linspace(0, time_vector[-1], 500)
x_interp = np.interp(time_interp, time_vector, x)
y_interp = np.interp(time_interp, time_vector, y)
z_interp = np.interp(time_interp, time_vector, z)  

#plot 3d trajectory
fig,ax = plt.subplots(1,1, subplot_kw={'projection': '3d'})
ax.plot3D(x_interp, y_interp, z_interp, 'gray')
fig.show()

inter_df = pd.DataFrame({'x':x_interp, 'y':y_interp, 'z':z_interp})
inter_df = data_handle.scale_cartesian_with_terrain(inter_df, grand_canyon)

interpolated_traj = go.Scatter3d(x=inter_df['x'],
                                 y=inter_df['y'],
                                 z=inter_df['z'],
                                 line=dict(
                                     color=time_interp,
                                     colorscale=colorscale,
                                     width=4,
                                    #  opacity=0.8,
                                 ),
                                 marker=dict(
                                     color=time_interp,
                                     colorscale=colorscale,
                                     size=3,
                                     opacity=0.3,
                                 ),
                                 name='planner trajectory')

#%% Compute trajectory of threats
overall_threat_trajectories = []
# for threat in threats:
#     for times in time_list:
#         threat_trajectory = []
#         for t in times:
#             position = threat.ellipse_trajectory(t)
#             threat_trajectory.append(position)
#         threat_trajectory = np.array(threat_trajectory)
        
#         #convert to dataframe 
#         threat_trajectory = pd.DataFrame({'x':threat_trajectory[:,0],
#                                             'y':threat_trajectory[:,1],
#                                             'z':threat_trajectory[:,2]})
        
#         #scale the trajectory
#         threat_trajectory = data_handle.scale_cartesian_with_terrain(threat_trajectory, 
#                                                                      grand_canyon)
        
#         overall_threat_trajectories.append(threat_trajectory)

for threat in threats:
    threat_trajectory = []
    
    for t in time_interp:
        position = threat.ellipse_trajectory(t)
        threat_trajectory.append(position)
    
    threat_trajectory = np.array(threat_trajectory)
    threat_trajectory = pd.DataFrame({'x':threat_trajectory[:,0],
                                      'y':threat_trajectory[:,1],
                                      'z':threat_trajectory[:,2]})
    threat_trajectory = data_handle.scale_cartesian_with_terrain(threat_trajectory,
                                                                 grand_canyon)

    overall_threat_trajectories.append(threat_trajectory)
        
        
#plot the threat trajectories as a function of time
fig,ax = plt.subplots(1,1, subplot_kw={'projection': '3d'})

#color scale for time
for threat in overall_threat_trajectories:
    #ax.plot3D(threat['x'], threat['y'], threat['z'], 'gray')
    ax.scatter3D(threat['x'], threat['y'], threat['z'], c=time_interp, cmap='viridis')
    
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

#show colorbar
fig.colorbar(ax.collections[0], ax=ax, label='time')
ax.legend()
fig.show()
   

#%% 

detect_info = detection_info_jit.items()
voxels = radar1.get_voxels(detect_info,100)
voxels_normalized = data_handle.format_radar_data_with_terrain(voxels, grand_canyon)
voxel_visualizer = radar1.get_visual_scatter_radar(voxels_normalized)
fig = grand_canyon.plot_3d_expanded(1, 1300 , 2200)

detect_info2 = detection_info_jit2.items()
voxels2 = radar2.get_voxels(detect_info2,100)
voxels_normalized2 = data_handle.format_radar_data_with_terrain(voxels2, grand_canyon)
voxel_visualizer2 = radar2.get_visual_scatter_radar(voxels_normalized2)

#plot the goal position
x_goal = goal_position.x/(grand_canyon.max_x - grand_canyon.min_x) * grand_canyon.expanded_array.shape[0]
y_goal = goal_position.y/(grand_canyon.max_y - grand_canyon.min_y) * grand_canyon.expanded_array.shape[1]
z_goal = goal_position.z

fig.add_trace(go.Scatter3d(x=[x_goal], y=[y_goal], z=[z_goal], 
                           mode='markers', 
                           name='Goal',
                           marker=dict(color='green', size=5)))

#plot the original position
x_pos = radar_pos.x/(grand_canyon.max_x - grand_canyon.min_x) * grand_canyon.expanded_array.shape[0]
y_pos = radar_pos.y/(grand_canyon.max_y - grand_canyon.min_y) * grand_canyon.expanded_array.shape[1]
z_pos = radar_pos.z

#plot the original position for radar 2
x_pos2 = radar_pos2.x/(grand_canyon.max_x - grand_canyon.min_x) * grand_canyon.expanded_array.shape[0]
y_pos2 = radar_pos2.y/(grand_canyon.max_y - grand_canyon.min_y) * grand_canyon.expanded_array.shape[1]
z_pos2 = radar_pos2.z

fig.add_trace(go.Scatter3d(x=[x_pos], y=[y_pos], z=[z_pos], 
                           name='Radar 1',
                           mode='markers', marker=dict(color='red', size=5)))
fig.add_trace(voxel_visualizer)

fig.add_trace(go.Scatter3d(x=[x_pos2], y=[y_pos2], z=[z_pos2],
                           name = 'Radar 2', 
                           mode='markers', marker=dict(color='red', size=5)))
fig.add_trace(voxel_visualizer2)

for scatter in scatter_list:
    fig.add_trace(scatter)

fig.add_trace(interpolated_traj)
    
#make another plot
regular_voxels = go.Scatter3d(x=voxels['voxel_x'], y=voxels['voxel_y'], 
                              z=voxels['voxel_z'],
                              mode='markers', marker=dict(color='blue', size=5))

regular_voxels2 = go.Scatter3d(x=voxels2['voxel_x'], y=voxels2['voxel_y'], 
                               z=voxels2['voxel_z'],
                              mode='markers', marker=dict(color='blue', size=5))

regular_position = go.Scatter3d(x=[radar_pos.x], y=[radar_pos.y], z=[radar_pos.z], 
                                mode='markers', 
                                marker=dict(color='red', size=5))
regular_position2 = go.Scatter3d(x=[radar_pos2.x], y=[radar_pos2.y], z=[radar_pos2.z], 
                                 mode='markers', 
                                marker=dict(color='red', size=5))

#plot threat trajectories
color_threats = ['red', 'black', 'red']
time_marker = dict(color=time_interp, 
                   colorscale=colorscale,
                   size = 3 , 
                   opacity=0.9,
                   colorbar=dict(thickness=20,
                                 x=-0.15,   
                                 title='Time (s)',))

for i, threat_trajectory in enumerate(overall_threat_trajectories):
    #color gradient as a function of time
    # fig.add_trace(go.Scatter3d(x=threat_trajectory['x'],
    #                            y=threat_trajectory['y'],
    #                            z=threat_trajectory['z'],
    #                            name = 'Dynamic Threat Trajectory',
    #                            mode='lines',
                               
    threat_name = 'Dynamic Threat Trajectory ' + str(i+1)
    line_threat = dict(color=color_threats[i], width=1, dash='dash',)
    fig.add_trace(go.Scatter3d(x=threat_trajectory['x'], 
                               y=threat_trajectory['y'], 
                               z=threat_trajectory['z'],
                               name= threat_name,
                            #    mode='lines', 
                                line=line_threat,
                               marker=time_marker,))


# # plot obstacles
# for obs in obs_list:
#     obs_position = obs.position
#     radius = obs.radius_m
#     #get elevation at the obstacle position
#     lat_dg, lon_dg = grand_canyon.latlon_from_cartesian(obs_position.x, obs_position.y)
#     elevation = grand_canyon.get_elevation_from_latlon(lat_dg, lon_dg)
#     base_height = elevation
#     elevated_z = base_height + obs_position.z
#     x, y, z = create_cylinder((obs_position.x, obs_position.y), elevated_z, elevation)
#     #scale x y z 
#     x = x/(grand_canyon.max_x - grand_canyon.min_x) * grand_canyon.expanded_array.shape[0]
#     y = y/(grand_canyon.max_y - grand_canyon.min_y) * grand_canyon.expanded_array.shape[1]
    
#     # z = z/(grand_canyon.max_z - grand_canyon.min_z) * grand_canyon.expanded_array.shape[2]
#     # df = pd.DataFrame({'x':x, 'y':y, 'z':z})
#     # df = data_handle.scale_cartesian_with_terrain(df, grand_canyon)
    
#     fig.add_surface(x=x, y=y, z=z, 
#                     colorscale='Greys', showscale=False, opacity=0.5)

#plot approach vector
x_vals = []
y_vals = []
z_vals = []
p_dense_vals = []

for path in best_approaches:
    for wp in path:
        x_vals.append(wp[0])
        y_vals.append(wp[1])
        z_vals.append(wp[2])
        p_dense_vals.append(wp[3])
        
data_handler = DataHandler()
info_dictionary = {'x': x_vals, 'y': y_vals, 'z': z_vals, 'p_dense': p_dense_vals}
formatted_df = pd.DataFrame(info_dictionary)
formatted_df = data_handler.scale_cartesian_with_terrain(
    formatted_df, grand_canyon)

# add approach vector 
aproach_vector_plot = go.Scatter3d(
    x=formatted_df['x'],
    y=formatted_df['y'],
    z=formatted_df['z'],
    mode='markers',
    name='voxel_data',
    marker=dict(
        color=p_dense_vals,
        colorscale='Viridis',
        # color_discrete_sequence=px.colors.qualitative.Plotly,
        size=3,
        opacity=0.1,
        colorbar=dict(
            title='Power Density',
            x=0)
    )
)

fig.add_trace(aproach_vector_plot)

#remove the colorbar
fig.update_layout(coloraxis_showscale=False)

# fig.update_traces(marker_showscale=False)
fig.show()

#save as html 
fig.write_html('Dynamic_Static_Threat_Approaches.html')

#save as png
fig.write_image('terrain_avoidance_radar.png')


# fig2.show()

# %%
