# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 10:40:37 2023

@author: mjb7tf
"""

import richdem as rd
import rasterio
from rasterio import plot
import geopandas as gpd
from shapely.geometry import Point
import math
from rasterio.windows import Window
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio

from src.Utils import measure_latlon_distance
from src.config.Config import utm_param

import pyproj
from pyproj import Proj, transform


pio.renderers.default='browser'

'''CLASS'''
class Terrain():
    """
    Utilizes UTMs to convert lat lon to cartesian coordinates
    
    
    """
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
        
        # self.min_x, self.min_y, self.min_z = self.cartesian_from_latlon(lat_min, lon_min) 
        # self.max_x, self.max_y, self.max_z = self.cartesian_from_latlon(lat_max, lon_max) 
        self.elevations        = self.generate_elevations()
        
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
        
        
    ''' USED BY GET ELEVATION FUNCTION TO GET THE LENGTH OF THE ARRAY'''
    def length_of_map(self):
        with rasterio.open(self.map_used) as src:
            window = src.window(self.lon_min, self.lat_min, self.lon_max, self.lat_max)
            area_of_intrest = src.read(1, window=window)
            aoi_len = len(area_of_intrest)
            return aoi_len
    
    '''RETURNS THE VALUE OF 1 SPECIFIC CELL IN THE ARRAY'''
    def cell_of_intrest(self, y, x):
        with rasterio.open(self.map_used) as src:
            window = src.window(self.lon_min, self.lat_min, self.lon_max, self.lat_max)
            area_of_intrest = src.read(1, window=window)
            value = area_of_intrest[y,x]
            return value
    
    '''PRINTS THE SELECTED AREA OF INTEREST'''
    def print_map(self):
        with rasterio.open(self.map_used) as src:
            window = src.window(self.lon_min, self.lat_min, self.lon_max, self.lat_max)
            area_of_intrest = src.read(1, window=window)
            rasterio.plot.show(area_of_intrest, cmap='bone')
           
    '''PRINTS THE ENTIRE MAP WITH COORDINATE LINES'''
    def print_whole_map(self):
        whole_map = rasterio.open(self.map_used)
        raster_width = whole_map.width
        raster_height = whole_map.height
        'width= {}, height={}'.format(raster_width,raster_height)
        rasterio.plot.show(whole_map, cmap='bone')
    
    '''SIMPLE MATH FUNC TO CHANGE FLOAT WITH 7 DECIMALS TO INT'''
    def truncate_float(self, number, decimal_places):
        multiplier = 10 ** decimal_places
        return int(number * multiplier) / multiplier
    
    '''GETS THE X COORDINATE OF A CELL WITHIN THE ARRAY'''
    def get_x_coor(self, x:float):
        x_range = self.truncate_float((self.lon_max - self.lon_min), 4)
        relative_x_pos = round(((x - self.lon_min) / x_range) * self.length_of_map())
        return relative_x_pos
        
    '''GETS THE Y COORDINATE OF A CELL WITHIN THE ARRAY'''
    def get_y_coor(self, y:float):
        y_range = self.truncate_float((self.lat_max - self.lat_min), 4)
        relative_y_pos = round(((y - self.lat_min) / y_range) * self.length_of_map())
        return relative_y_pos
    
    '''RETURNS THE ELEVATION OF ANY GIVEN EARTH COORDINATES WITHIN THE ARRAY'''
    def get_elevation(self, x:float, y:float):
        """
        This x and y are the lat and lon of the point of intrest
        """
        x_pos = self.get_x_coor(x)
        y_pos = self.get_y_coor(y)
        
        print("The x cell is "+str(x_pos))
        print("The y cell is "+str(y_pos))
        
        print("The cell value is " + str(self.cell_of_intrest(y_pos, x_pos)) + "m")
    
    '''PLOTS THE AREA OF INTREST IN 3D USING MATPLOTLIB'''
    def plot_3d_matlib(self):
        with rasterio.open(self.map_used) as src:
            window = src.window(self.lon_min, self.lat_min, self.lon_max, self.lat_max)
            area_of_intrest = src.read(1, window=window)
        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        
        x = np.arange(0, self.length_of_map(), 1)
        y = np.arange(0, self.length_of_map(), 1)
        x, y = np.meshgrid(x,y)
        z = area_of_intrest[y,x]
        
        
        '''PLOT THE SURFACE'''
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

    def generate_elevations(self):
        with rasterio.open(self.map_used) as src:
            window = src.window(
                self.lon_min, self.lat_min, self.lon_max, self.lat_max)
            area_of_intrest = src.read(1, window=window)
        
        x1 = np.arange(0, self.length_of_map(), 1)
        y1 = np.arange(0, self.length_of_map(), 1)
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

        return result_array
    
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
        
    def cartesian_from_latlon(
        self, lat:float, lon:float, include_bias:bool=False) -> tuple:
        """
        returns the cartesian coordinates of a given lat lon point
        if include_bias is true, then the bias is included in the output
        """ 
        R = 6371.0 # Radius of earth in KM
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)

        # Convert to Cartesian coordinates
        x = R * math.cos(lat_rad) * math.cos(lon_rad)
        y = R * math.cos(lat_rad) * math.sin(lon_rad)
        z = R * math.sin(lat_rad)
        
        if include_bias ==  True:
            x = x - self.min_x
            y = y - self.min_y
            z = z - self.min_z
        
        #convert to meters
        x = x * 1000
        y = y * 1000
        z = z * 1000
            
        return x, y, z
    
    def latlon_from_cartesian(
        self, x:float, y:float, z:float, include_bias:bool=False) -> tuple:
        """
        Convert cartesian coordinates to lat lon coordinates in degrees
        """

        lat_rad = math.atan2(z, math.sqrt(x**2 + y**2))
        lon_rad = math.atan2(y, x)
        
        lat_dg = math.degrees(lat_rad)
        lon_dg = math.degrees(lon_rad)

        # if include_bias ==  True:
        #     lat = lat + self.lat_min
        #     lon = lon + self.lon_min
            
        return lat_dg, lon_dg
    
    
'''INSTANCES'''
# grand_canyon = Terrain('tif_data/n36_w113_1arc_v3.tif', 
#                        lon_min = -112.55, 
#                        lon_max = -112.4, 
#                        lat_min = 36.2, 
#                        lat_max = 36.35)

#has to be a square
grand_canyon = Terrain('tif_data/n36_w113_1arc_v3.tif', 
                       lon_min = -112.5, 
                       lon_max = -112.45, 
                       lat_min = 36.2, 
                       lat_max = 36.25,
                       utm_zone=utm_param['grand_canyon'])


grand_canyon.print_information()

min_x = grand_canyon.min_x
min_y = grand_canyon.min_y

max_x = grand_canyon.max_x
max_y = grand_canyon.max_y

distance = np.linalg.norm(np.array([min_x, min_y]) - np.array([max_x, max_y]))
# distance = measure_latlon_distance(
#     lat1=lat_lon[0], lon1=lat_lon[1], lat2=max_lat_lon[0], lon2=max_lat_lon[1])
print("The distance between the min and max lat lon is " + str(distance) + "m")

dx = max_x - min_x
dy = max_y - min_y


print("The dx is " + str(dx) + "m")
print("The dy is " + str(dy) + "m")

# cartesian_point = grand_canyon.cartesian_from_latlon()
# print(cartesian_point)

'''
lgtf = Terrain(map_used = 'sullivan_indiana.tif', 
                       x_min = -87.375,
                       x_max = -87.335, 
                       y_min = 39.105, 
                       y_max = 39.145)
'''

# lgtf = Terrain(map_used = 'tif_data/sullivan_indiana.tif', 
#                        x_min = -87.375,
#                        x_max = -87.345, 
#                        y_min = 39.105, 
#                        y_max = 39.135)

# kansas_city = Terrain('tif_data/kansas_city.tif', 
#                       x_min = -95, 
#                       x_max = -94.95, 
#                       y_min = 39, 
#                       y_max = 39.05)


'''EXECUTION'''
# grand_canyon.get_elevation( -112.5 , 36.3)
# grand_canyon.print_map()
# grand_canyon.print_whole_map()
# grand_canyon.plot_3d_matlib()
grand_canyon.plot_3d_expanded(3 , 0 , 16000)


# lgtf.plot_3d_expanded(15 , 0 , 1600)
# lgtf.print_map()
# lgtf.plot_3d_matlib()

# kansas_city.print_whole_map()
# kansas_city.plot_3d_expanded(3)
