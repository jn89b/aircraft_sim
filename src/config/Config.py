import numpy as np


OBSTACLE_AVOID = False
MULTIPLE_OBSTACLE_AVOID = False
RADAR_AVOID = False
N_OBSTACLES = 0
BUFFER_DISTANCE = 0.0

ACCEL_LIM = 14 * 9.81
MAX_RADIAN = np.deg2rad(2000) 

HOVER_THROTTLE = 0.5

#NLP solver options
MAX_ITER = 1500
MAX_TIME = 0.1
PRINT_LEVEL = 0
ACCEPT_TOL = 1e-2
ACCEPT_OBJ_TOL = 1e-2   
PRINT_TIME = 0

#Gravity    
G = 9.81
RHO = 1.225

#Target options
TARGET_DISCHARGE_RATE = 0.1


utm_param = {
    'grand_canyon': 'espg:32612',
    'kansas_city': 'espg:32614',
    'new_york': 'espg:32618',
    'indiana': 'espg:32616',
}
