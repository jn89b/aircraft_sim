import numpy as np


class SentryThreats():
    """
    """    
    def __init__(self, init_position:np.ndarray,
                 prev_position:np.ndarray,
                 velocity:float, use_ellipse:bool=True,
                 ellipse_params:dict=None) -> None:
        
        self.position = init_position
        self.prev_position = init_position
        self.velocity = velocity
        
        if use_ellipse:
            self.use_ellipse = True
            self.ellipse_params = ellipse_params

    def ellipse_trajectory(self, t:float) -> np.ndarray:
        """
        """    
        a = self.ellipse_params['a']
        b = self.ellipse_params['b']
    
        delta_position = self.position - self.prev_position
        psi = np.arctan2(delta_position[1], delta_position[0]) 
        R_ellipse = ((a**2 * np.sin(psi)**2) + (b**2 * np.cos(psi)**2))**1.5 / (a*b) 
          
        x = (self.position[0])  + (a * np.cos(t))
        y = (self.position[1])  + (b * np.sin(t))      
        z = self.position[2]
        
        return np.array([x, y, z])

    
    def update_position(self, t:float) -> np.ndarray:        
        if self.use_ellipse:
            self.position = self.ellipse_trajectory(t)
            return self.position
        else:
            return self.init_position + self.velocity * t

        
ellipse_params = {'a': 1, 'b': 1}
velocity = 1
init_position = np.array([0, 0, 0])
prev_position = np.array([0, 0, 0])

sentry = SentryThreats(init_position, prev_position, velocity, 
                       use_ellipse=True, ellipse_params=ellipse_params)

position_history = []

t_max = 1000
t_init = 0

dt = 0.1

for i in range(int(t_max/dt)):
    t = t_init + i*dt
    position_history.append(sentry.update_position(t))

position_history = np.array(position_history)
x = position_history[:, 0]
y = position_history[:, 1]

import matplotlib.pyplot as plt
plt.plot(x, y)
plt.show()

