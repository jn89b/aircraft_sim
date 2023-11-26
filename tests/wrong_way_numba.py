"""
Using numba troubleshooting
"""

from numba import jit, njit, vectorize
import time 
import numpy as np

def original_function(input_list:list):
    output_list = []
    for item in input_list:
        if item % 2 == 0:
            output_list.append(item)
        else:
            output_list.append('1')
            
    return output_list

@vectorize #works on scalars and arrays
def scalar_function(item):
    if item % 2 == 0:
        return item
    else:
        return 1

n_vals = 1000000
some_list = np.array(range(n_vals))



        