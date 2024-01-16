import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import seaborn as sns
import pickle as pkl

import plotly.graph_objects as go

from src.guidance_lib.src.Terrain import Terrain
from src.guidance_lib.src.PositionVector import PositionVector
from src.guidance_lib.src.Grid import Grid, FWAgent
from src.guidance_lib.src.Radar import Radar
from src.guidance_lib.src.Obstacle import Obstacle
from src.guidance_lib.src.SparseAstar import SparseAstar
from src.config.Config import utm_param
from src.data_vis.DataParser import DataHandler
import random


data_handler = DataHandler()
# start_position = PositionVector(10, 60, 2000)
# goal_position  = PositionVector(5000, 5000, 1500)

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

start_lat, start_lon = grand_canyon.latlon_from_cartesian(start_position.x, 
                                                          start_position.y)
goal_lat, goal_lon = grand_canyon.latlon_from_cartesian(goal_position.x,
                                                        goal_position.y)

start_elevation = grand_canyon.get_elevation_from_latlon(start_lat,start_lon)
goal_elevation  = grand_canyon.get_elevation_from_latlon(goal_lat, goal_lon)

print('start elevation: ', start_elevation)
print('goal elevation: ', goal_elevation)
                                                        
sparse_astar = SparseAstar(grid=grid, terrain_map=grand_canyon, 
                           use_terrain=True, velocity=15,
                           terrain_buffer_m=10)

sparse_astar.init_nodes()
path = sparse_astar.search()

# planner_states = return_planner_states(sparse_astar, path)
planner_states = data_handler.return_planner_states(path)

#save to csv
planner_states.to_csv('planner_states.csv')

formatted_states = data_handler.scale_cartesian_with_terrain(planner_states, 
                                                         grand_canyon)

#plot in 2D
fig, ax = plt.subplots()
ax.plot(planner_states['x'], planner_states['y'], 'o-')
ax.set_xlabel('X')
ax.set_ylabel('Y')

fig = grand_canyon.plot_3d_expanded(1, 0 , 2000)

# fig = grand_canyon.plot_3d()
trajectory = go.Scatter3d(x=formatted_states['x'], 
                          y=formatted_states['y'], 
                          z=planner_states['z'],
                          marker=dict(
                              size=10,
                              color='blue',
                              opacity=0.5
                            #   color=planner_states['z'],
                            #   colorscale='Viridis',
                          ),
                          name='planner trajectory')

#add on to figure
fig.add_trace(trajectory)
fig.show()


