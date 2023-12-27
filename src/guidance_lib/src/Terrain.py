
import numpy as np
import matplotlib.pyplot as plt
import rasterio

import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio


import pyproj
from pyproj import Proj, transform, Transformer

import time 

pio.renderers.default='browser'

class Terrain():
    def __init__(self, map_used:str, lon_min:float , lon_max:float, 
                 lat_min:float, lat_max:float,
                 utm_zone:str = 'espg:32612'):
        self.map_used = map_used
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_min = lat_min
        self.lat_max = lat_max
        
        self.origin_latlon = (self.lat_min, self.lon_min)
        self.max_latlon = (self.lat_max, self.lon_max)
        
        self.wgs84 = pyproj.Proj(init='epsg:4326')
        
        ## this is stupid but you have to hard code the utm zone
        self.utm_zone = pyproj.Proj(init='epsg:32612')
        
        self.min_x, self.min_y = pyproj.transform(
            self.wgs84, self.utm_zone, self.lon_min, self.lat_min)
        
        self.max_x , self.max_y = pyproj.transform(
            self.wgs84, self.utm_zone, self.lon_max, self.lat_max)

        #this is also stupid I had to hardcode this
        self.transformer = Transformer.from_crs(
            32612, 4326, always_xy=True)
        
        with rasterio.open(self.map_used) as src:
            self.src = src
            self.elevations = src.read(1)
            
    def get_elevation_from_latlon(self, lon_dg:float, lat_dg:float) -> float:
        """returns the elevation of a given lat lon point"""

        idx = self.src.index(lon_dg, lat_dg) 
        
        #check if idx is in the bounds of the map
        if idx[0] < 0 or idx[0] >= len(self.elevations) or \
            idx[1] < 0 or idx[1] >= len(self.elevations[0]):
            
                return None       
        
        return self.elevations[idx]

    def cartesian_from_latlon(self,
                                lat_dg:float,
                                lon_dg:float,
                                include_bias:bool=False) -> tuple:
            """
            returns the cartesian coordinates of a given lat_dg lon_dg point
            """        
            x,y = self.transformer.transform(
                lon_dg, lat_dg)
            
            return x, y
        
    def latlon_from_cartesian(self,
                                x:float,
                                y:float,
                                include_bias:bool=True) -> tuple:
            """
            Convert cartesian coordinates to lat lon coordinates in degrees
            """
            if include_bias ==  True:
                x = x + self.min_x
                y = y + self.min_y
                
            lon_dg, lat_dg = self.transformer.transform(
                x,y)
            
            return lat_dg, lon_dg
        
    def print_information(self) -> None:
        print("The map used is " + self.map_used)
        print("The minimum longitude is " + str(self.lon_min))
        print("The maximum longitude is " + str(self.lon_max))
        print("The minimum latitude is " + str(self.lat_min))
        print("The maximum latitude is " + str(self.lat_max))
        
        print("The origin lat lon is " + str(self.origin_latlon))
        print("The max lat lon is " + str(self.max_latlon))
        
        print("The min x is " + str(self.min_x))
        print("The min y is " + str(self.min_y))
        print("The max x is " + str(self.max_x))
        print("The max y is " + str(self.max_y))
        
        
    '''PRINTS THE ENTIRE MAP WITH COORDINATE LINES'''
    def print_whole_map(self):
        whole_map = rasterio.open(self.map_used)
        raster_width = whole_map.width
        raster_height = whole_map.height
        'width= {}, height={}'.format(raster_width,raster_height)
        rasterio.plot.show(whole_map, cmap='bone')
        
    '''PLOTS IN 3D USING PLOTLY'''
    def plot_3d_expanded(self, step_number:int, z_min:float , z_max:float):
        z1 = self.generate_elevations()
        expanded_array = self.expand_array(z1, step_number)
        x1 = np.arange(0, expanded_array.shape[0], 1)
        y1 = np.arange(0, expanded_array.shape[1], 1)
        
        surface = go.Surface(z = expanded_array, x = x1, y= y1)
        
        # Customize the z-axis limits
        print("min elevation is", np.min(expanded_array))
        # z_min = 0
        # z_max = 10000
        # surface.update_zaxes(range=[z_min, z_max])
        fig = go.Figure(data = [surface])
        
        
        fig.update_layout(
            scene = dict(zaxis = dict(nticks=4, range=[z_min, z_max])))
                
        fig.show()