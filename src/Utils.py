import pandas as pd
import numpy as np


# Use as utility
def get_airplane_params(df:pd.DataFrame) -> dict:
    airplane_params = {}
    for index, row in df.iterrows():
        #check if we can't convert to float
        try:
            float(row["Value"])
        except:
            continue
        airplane_params[row["Variable"]] = float(row["Value"])
    
    return airplane_params

def read_lon_matrices(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    matrix_a_lines = lines[2:6]
    matrix_b_lines = lines[9:]

    matrix_a = np.array([list(map(float, line.split())) for line in matrix_a_lines])
    matrix_b = np.array([list(map(float, line.split())) for line in matrix_b_lines])

    #remove the first list
    matrix_b = matrix_b[1:]
    
    #create a 4 x 2 matrix from matix  b
    B = np.zeros((4,2))
    for i in range(len(matrix_b)):
        print(matrix_b[i])
        B[i] = matrix_b[i]  

    return matrix_a, B

def measure_latlon_distance(lat1:float, lon1:float, 
                     lat2:float, lon2:float, 
                     x_offset:float=0, 
                     y_offset:float=0) -> tuple:
    """
    returns the distance between two lat lon points in meters
    """
    R = 6378.137 # Radius of earth in KM
    d_lat = lat2 * np.pi / 180 - lat1 * np.pi / 180
    d_lon = lon2 * np.pi / 180 - lon1 * np.pi / 180
    a = np.sin(d_lat/2) * np.sin(d_lat/2) + np.cos(lat1 * np.pi / 180) * np.cos(lat2 * np.pi / 180) * np.sin(d_lon/2) * np.sin(d_lon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c * 1000

    return d
    
