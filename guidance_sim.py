import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import seaborn as sns
import pickle as pkl

# how to refer to modules in src
from src.guidance_lib.src.PositionVector import PositionVector
from src.guidance_lib.src.Grid import Grid, FWAgent
from src.guidance_lib.src.Radar import Radar
from src.guidance_lib.src.Obstacle import Obstacle
from src.guidance_lib.src.SparseAstar import SparseAstar
from src.guidance_lib.src.Config.radar_config import RADAR_AIRCRAFT_HASH_FILE
from src.guidance_lib.src.DataContainer import SimDataContainer
from src.guidance_lib.src.SentryThreats import SentryThreats

"""
To do 
Make this small scale first

Create a simple c space with one radar
Have some obstacles placed within range
"""

import plotly.graph_objects as go
import plotly.express as px
import pickle as pkl
import pandas as pd

sns.set_palette("colorblind")

def load_pickle():
    """
    pass
    """
    with open('radar_params_obs.pickle', 'rb') as file:
        loaded_data = pkl.load(file)

    return loaded_data

def return_planner_states(self, states:list) ->pd.DataFrame:
    x = [state[0] for state in states]
    y = [state[1] for state in states]
    z = [state[2] for state in states]
    theta_dg = [state[3] for state in states]
    phi_dg = [state[4] for state in states]
    psi_dg = [state[5] for state in states]

    #return a dataframe
    planner_states = pd.DataFrame({'x':x, 'y':y, 'z':z, 
                                   'theta_dg':theta_dg, 
                                   'phi_dg':phi_dg, 
                                   'psi_dg':psi_dg})

    return planner_states

if __name__ == '__main__':

    #load simple sim
    start_position = PositionVector(10,10,0)
    goal_position = PositionVector(400,450,20)
    fw_agent_psi_dg = 25
    fw_agent = FWAgent(start_position, 0, fw_agent_psi_dg)
    fw_agent.vehicle_constraints(horizontal_min_radius_m=60, 
                                 max_climb_angle_dg=5,
                                 max_psi_turn_dg=25)
    fw_agent.leg_m = 25

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

    planner_states = return_planner_states(sparse_astar, path)
    print(planner_states)
    #save planner 
    planner_states.to_csv("planner_states.csv", index=False)

    #plot in 2D
    fig, ax = plt.subplots()
    ax.plot(planner_states['x'], planner_states['y'], 'o-')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    #plot obstacles
    for obs in obs_list:
        circle = plt.Circle((obs.position.x, obs.position.y), obs.radius_m, color='r')
        ax.add_artist(circle)

    plt.show()
    

    