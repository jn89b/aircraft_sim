# Function to generate cylinder data
import numpy as np

def create_cylinder(center, height, radius, base_height=1500):
    x0, y0 = center
    z = np.linspace(base_height, height, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + x0
    y_grid = radius * np.sin(theta_grid) + y0
    return x_grid, y_grid, z_grid