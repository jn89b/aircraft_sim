
import numpy as np
import matplotlib.pyplot as plt
import rasterio

import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio
import pyproj

from rasterio import plot
from rasterio.windows import from_bounds
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
            
        z1 = self.generate_elevations()
        z1 = np.array([row[::-1] for row in z1])
        self.expanded_array = self.expand_array(z1, 1)
            
    def get_elevation_from_latlon(self, lat_dg:float, lon_dg:float) -> float:
        """returns the elevation of a given lat lon point"""

        idx = self.src.index(lon_dg, lat_dg) 
        
        #check if idx is in the bounds of the map
        # if idx[0] < 0 or idx[0] >= len(self.elevations) or \
        #     idx[1] < 0 or idx[1] >= len(self.elevations[0]):
            
        #         return None       
        
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
        
    ''' USED BY GET ELEVATION FUNCTION TO GET THE LENGTH OF THE ARRAY'''
    def length_of_map(self):
        with rasterio.open(self.map_used) as src:
            # window = src.window(
            #     self.lon_min, self.lat_min, self.lon_max, self.lat_max)
            
            bounds = from_bounds(left=self.lon_min, bottom=self.lat_min,
                                 right=self.lon_max, top=self.lat_max,
                                 transform=src.transform)
            area_of_intrest = src.read(1, window=bounds)
            aoi_len = len(area_of_intrest)
            return aoi_len
        
    def generate_elevations(self):
        with rasterio.open(self.map_used) as src:
            bounds = from_bounds(left=self.lon_min, bottom=self.lat_min,
                                 right=self.lon_max, top=self.lat_max,
                                 transform=src.transform)
            
            area_of_intrest = src.read(1, window=bounds)
        
        map_length = self.length_of_map()
        x1 = np.arange(0, map_length, 1)
        y1 = np.arange(0, map_length, 1)
        x1, y1 = np.meshgrid(x1,y1)
        z1 = area_of_intrest[x1,y1]
        return z1
    
    def expand_array(self, z1:np.ndarray, size_array:int):
        num_rows = len(z1) * size_array
        num_cols = len(z1[0]) * size_array

        # Create an empty array of zeros with the new dimensions
        result_array = np.zeros((num_rows, num_cols), dtype=int)

        # Populate the new array with values from the original array
        for i in range(num_rows):
            for j in range(num_cols):
                original_row = i // size_array  # Determine the corresponding row in the original array
                original_col = j // size_array  # Determine the corresponding column in the original array
                result_array[i, j] = z1[original_row][original_col]

        #reverse the array so that the origin is in the bottom left
        
        return result_array
        
    '''PLOTS IN 3D USING PLOTLY'''
    def plot_3d_expanded(self, step_number:int, z_min:float , z_max:float) -> go.Figure:
        z1 = self.generate_elevations()
        # z1 = z1[::-1]
        z1 = np.array([row[::-1] for row in z1])
        self.expanded_array = self.expand_array(z1, step_number)
        x1 = np.arange(0, self.expanded_array.shape[0], 1)
        y1 = np.arange(0, self.expanded_array.shape[1], 1)
        surface = go.Surface(z = self.expanded_array, x = x1, y= y1, showscale=False)
        
        # Customize the z-axis limits
        # z_min = 0
        # z_max = 10000
        # surface.update_zaxes(range=[z_min, z_max])
        fig = go.Figure(data = [surface])
        
        fig.update_layout(
            scene = dict(zaxis = dict(nticks=4, range=[z_min, z_max])))
        
        # fig.update(layout_coloraxis_showscale=False)
                
        # fig.show()
        return fig
    
    def plot_3d(self):
        x_range = np.arange(self.min_x, self.max_x, 1)
        y_range = np.arange(self.min_y, self.max_y, 1)
        # elevations = self.elevations
        elevation_array = np.zeros((len(x_range), len(y_range)), dtype=int)        

        for i in range(len(x_range)):
            for j in range(len(y_range)):
                lat, lon = self.latlon_from_cartesian(x_range[i],y_range[j],False)
                z = self.get_elevation_from_latlon(lat, lon)
                elevation_array[i][j] = z
                
        self.elevation_array = elevation_array
        x,y = np.meshgrid(x_range,y_range)
        z = elevation_array[x,y]
        
        surface = go.Surface(z = z, x = x, y= y)
        #set opacity of surface
        surface.update_traces(opacity=0.5)
        fig = go.Figure(data = [surface])
            
        return fig