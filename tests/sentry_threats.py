import numpy as np

# https://math.stackexchange.com/questions/426150/what-is-the-general-equation-of-the-ellipse-that-is-not-in-the-origin-and-rotate

class SentryThreats():
    """
    """    
    def __init__(self, init_position:np.ndarray,
                 prev_position:np.ndarray,
                 velocity:float, 
                 heading_rad:float=0,
                 use_ellipse:bool=True,
                 ellipse_params:dict=None,
                 radar_params:dict=None) -> None:
        
        self.position = init_position
        self.prev_position = prev_position
        self.velocity = velocity
        self.radar_params = radar_params
        self.heading_rad = heading_rad
        
        if use_ellipse:
            self.use_ellipse = True
            self.ellipse_params = ellipse_params
            
    def ellipse_trajectory(self, t:float) -> np.ndarray:
        x = self.ellipse_params['a'] * np.cos(t)
        y = self.ellipse_params['b'] * np.sin(t)
        z = self.position[2]
        self.prev_position = self.position
    
        return np.array([x, y, z])

    def update_position(self, t:float) -> np.ndarray:        
        if self.use_ellipse:
            self.position = self.ellipse_trajectory(t)
            self.heading_rad = np.arctan2(self.position[1] - self.prev_position[1],
                                            self.position[0] - self.prev_position[0])
            return self.position
        else:
            return self.init_position + self.velocity * t

    def is_inside_radar(self, x:float, y:float, z:float) -> bool:
        radar_range_a = self.radar_params['range_a']
        dx = x - self.position[0]
        dy = y - self.position[1]
        dz = z - self.position[2]
        
        if dx**2 + dy**2 + dz**2 <= radar_range_a**2:
            return True
        
        return False
        
        
ellipse_params = {'a': 5, 'b': 2}
radar_params = {'range_a': 1, 'range_b': 1}
velocity = 1
init_position = np.array([0, 0, 0])
prev_position = np.array([0, 0, 0])
heading_rad = 0

sentry = SentryThreats(init_position, prev_position, velocity, heading_rad, 
                       use_ellipse=True, ellipse_params=ellipse_params, 
                       radar_params=radar_params)

position_history = []
rotated_radar = []
t_max = 10
t_init = 0

dt = 0.1

test_position = np.array([1, 1, 0])

for i in range(int(t_max/dt)):
    t = t_init + i*dt
    position_history.append(sentry.update_position(t))
    heading_dg = np.rad2deg(sentry.heading_rad)
    print(f'Heading: {heading_dg}')
    
    if sentry.is_inside_radar(test_position[0], test_position[1], test_position[2]):
        print(f'Inside radar: {test_position}')
        rotated_radar.append(test_position)
        break


position_history = np.array(position_history)
radar_history = np.array(rotated_radar)
x = position_history[:, 0]
y = position_history[:, 1]

import matplotlib.pyplot as plt
plt.scatter(x, y)

for i in range(len(radar_history)):
    plt.plot(radar_history[i, 0], radar_history[i, 1], 'ro')

plt.legend()
plt.show()

