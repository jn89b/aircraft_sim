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

