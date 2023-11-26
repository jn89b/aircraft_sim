from numba import jit, njit, vectorize
import numpy as np
import time 
import sys

# from src.guidance_lib.src.PositionVector import PositionVector


def fast_voxel_algo3D(x0:float, y0:float, z0:float, 
                      x1:float, y1:float, z1:float, 
                      obs_list=[]) ->list:
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)

    x = int(np.floor(x0))
    y = int(np.floor(y0))
    z = int(np.floor(z0))

    if dx == 0:
        dt_dx = 1000000
    else:
        dt_dx = 1/ dx
    
    if dy == 0:
        dt_dy = 1000000
    else:
        dt_dy = 1/ dy
    
    if dz == 0:
        dt_dz = 1000000
    else:
        dt_dz = 1/ dz

    t = 0
    n = 1 
    t_next_horizontal = 0
    t_next_vertical = 0
    t_next_height = 0

    if (dx == 0):
        x_inc = 0
        t_next_horizontal = dt_dx 
    elif (x1 > x0):
        x_inc = 1
        n += int(np.floor(x1)) - x
        t_next_horizontal = (np.floor(x0) + 1 - x0) * dt_dx
    else:
        x_inc = -1
        n += x - int(np.floor(x1))
        t_next_horizontal = (x0 - np.floor(x0)) * dt_dx

    if (dy == 0):
        y_inc = 0
        t_next_vertical = dt_dy 
    elif (y1 > y0):
        y_inc = 1
        n += int(np.floor(y1)) - y
        t_next_vertical = (np.floor(y0) + 1 - y0) * dt_dy
    else:
        y_inc = -1
        n += y - int(np.floor(y1))
        t_next_vertical = (y0 - np.floor(y0)) * dt_dy

    if (dz == 0):
        z_inc = 0
        t_next_height = dt_dz 
    elif (z1 > z0):
        z_inc = 1
        n += int(np.floor(z1)) - z
        t_next_height = (np.floor(z0) + 1 - z0) * dt_dz
    else:
        z_inc = -1
        n += z - int(np.floor(z1))
        t_next_height = (z0 - np.floor(z0)) * dt_dz


    cell_rays = []
    for i in range(n):        
        # if obs_list:
        #     for obs in obs_list:
        #         pos = PositionVector(x,y)
        #         if obs.is_inside2D(pos,0.0) == True:
        #             return cell_rays

        cell_rays.append((x,y,z))
        # dy < dx then I need to go up since 
        if (t_next_horizontal < t_next_vertical):
            if (t_next_horizontal < t_next_height):
                x += x_inc
                t = t_next_horizontal
                t_next_horizontal += dt_dx
            else:
                z += z_inc
                t = t_next_height
                t_next_height += dt_dz
        else:
            if (t_next_vertical < t_next_height):
                y += y_inc
                t = t_next_vertical
                t_next_vertical += dt_dz
            else:
                z += z_inc
                t = t_next_height
                t_next_height += dt_dz
            
        # print(n)
    return cell_rays

@njit
def fast_voxel_algo3D_jit(x0:float, y0:float, z0:float, 
                      x1:float, y1:float, z1:float) ->list:
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)

    x = int(np.floor(x0))
    y = int(np.floor(y0))
    z = int(np.floor(z0))

    if dx == 0:
        dt_dx = 1000000
    else:
        dt_dx = 1/ dx
    
    if dy == 0:
        dt_dy = 1000000
    else:
        dt_dy = 1/ dy
    
    if dz == 0:
        dt_dz = 1000000
    else:
        dt_dz = 1/ dz

    t = 0
    n = 1 
    t_next_horizontal = 0
    t_next_vertical = 0
    t_next_height = 0

    if (dx == 0):
        x_inc = 0
        t_next_horizontal = dt_dx 
    elif (x1 > x0):
        x_inc = 1
        n += int(np.floor(x1)) - x
        t_next_horizontal = (np.floor(x0) + 1 - x0) * dt_dx
    else:
        x_inc = -1
        n += x - int(np.floor(x1))
        t_next_horizontal = (x0 - np.floor(x0)) * dt_dx

    if (dy == 0):
        y_inc = 0
        t_next_vertical = dt_dy 
    elif (y1 > y0):
        y_inc = 1
        n += int(np.floor(y1)) - y
        t_next_vertical = (np.floor(y0) + 1 - y0) * dt_dy
    else:
        y_inc = -1
        n += y - int(np.floor(y1))
        t_next_vertical = (y0 - np.floor(y0)) * dt_dy

    if (dz == 0):
        z_inc = 0
        t_next_height = dt_dz 
    elif (z1 > z0):
        z_inc = 1
        n += int(np.floor(z1)) - z
        t_next_height = (np.floor(z0) + 1 - z0) * dt_dz
    else:
        z_inc = -1
        n += z - int(np.floor(z1))
        t_next_height = (z0 - np.floor(z0)) * dt_dz


    # cell_rays = []
    #empty numpy array
    cell_rays = np.empty((n,3))
    for i in range(n):        
        # if obs_list:
        #     for obs in obs_list:
        #         pos = PositionVector(x,y)
        #         if obs.is_inside2D(pos,0.0) == True:
        #             return cell_rays

        # cell_rays.append((x,y,z))
        cell_rays[i,0] = x
        cell_rays[i,1] = y
        cell_rays[i,2] = z
        
        # dy < dx then I need to go up since 
        if (t_next_horizontal < t_next_vertical):
            if (t_next_horizontal < t_next_height):
                x += x_inc
                t = t_next_horizontal
                t_next_horizontal += dt_dx
            else:
                z += z_inc
                t = t_next_height
                t_next_height += dt_dz
        else:
            if (t_next_vertical < t_next_height):
                y += y_inc
                t = t_next_vertical
                t_next_vertical += dt_dz
            else:
                z += z_inc
                t = t_next_height
                t_next_height += dt_dz
            
        # print(n)
    return cell_rays


x_0 = 0
y_0 = 0
z_0 = 0

x_end = 5000
y_end = 5000
z_end = 5000


start_time = time.time()
cell_rays_no_jit = fast_voxel_algo3D(x_0, y_0, z_0, x_end, y_end, z_end)
print("--- %s seconds with no jit ---" % (time.time() - start_time))

#precompile with jit
x_end = 10
y_end = 10
z_end = 10
cell_rays_jit = fast_voxel_algo3D_jit(x_0, y_0, z_0, x_end, y_end, z_end)   


x_end = 1000
y_end = 1000
z_end = 1000
start_time = time.time()
cell_rays_jit = fast_voxel_algo3D_jit(x_0, y_0, z_0, x_end, y_end, z_end)   
print("--- %s seconds with jit ---" % (time.time() - start_time))


#plot the results with plotly
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

#convert to pandas dataframe
cell_rays_no_jit_df = pd.DataFrame(cell_rays_no_jit, columns=['x','y','z'])

fig = px.scatter_3d(cell_rays_no_jit_df, x='x', y='y', z='z')
fig.show()