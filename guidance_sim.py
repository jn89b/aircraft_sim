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

"""
To do 
Make this small scale first

Create a simple c space with one radar
Have some obstacles placed within range

"""

import plotly.graph_objects as go
import plotly.express as px
import pickle as pkl


sns.set_palette("colorblind")

def load_pickle():
    """
    pass
    """
    with open('radar_params_obs.pickle', 'rb') as file:
        loaded_data = pkl.load(file)

    return loaded_data


if __name__ == '__main__':

    #load simple sim
    

    

    
    