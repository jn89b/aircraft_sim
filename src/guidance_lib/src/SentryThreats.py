import numpy as np
from src.guidance_lib.src.Radar import Radar

class SentryThreats():
    """
    """    
    def __init__(self, init_position:np.ndarray,
                 prev_position:np.ndarray,
                 velocity_m:float, 
                 heading_rad:float=0,
                 use_ellipse:bool=True,
                 ellipse_params:dict=None,
                 radar_params:dict=None) -> None:
        
        self.position      = init_position
        self.prev_position = prev_position
        self.velocity_m    = velocity_m
        self.radar_params  = radar_params
        self.heading_rad   = heading_rad
        self.radar         = Radar(radar_params)
        
        if use_ellipse:
            self.use_ellipse = True
            self.ellipse_params = ellipse_params
            
    def ellipse_trajectory(self, t:float) -> np.ndarray:
        x = self.position[0] + self.ellipse_params['a'] * np.cos(t)
        y = self.position[1] + self.ellipse_params['b'] * np.sin(t)
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
            return self.position + self.velocity_m * t

    def is_inside_radar(self, x:float, y:float, z:float) -> bool:
        radar_range_m = self.radar_params['radar_range_m']
        dx = x - self.position[0]
        dy = y - self.position[1]
        dz = z - self.position[2]
        
        if dx**2 + dy**2 + dz**2 <= radar_range_m**2:
            return True
        
        return False
    
    def compute_prob_detect(self, dist:float, rcs_val:float, 
                            is_linear_db:bool=False) -> float:
        return self.radar.compute_prob_detect(dist, rcs_val, is_linear_db)