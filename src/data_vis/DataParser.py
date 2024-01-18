import pandas as pd

from src.guidance_lib.src.Terrain import Terrain

class DataHandler():
    """
    This is mostly a utility class to handle formatting of data
    
    """
    
    def __init__(self) -> None:
        pass
    
    @staticmethod    
    def return_planner_states(states:list) ->pd.DataFrame:
        """
        This function takes in a list of states and returns a dataframe
        input: list of states from sparse astar
        output: dataframe of states
        where each state is a list of [x, y, z, theta_dg, phi_dg, psi_dg]
        """
        x = [state[0] for state in states]
        y = [state[1] for state in states]
        z = [state[2] for state in states]
        theta_dg = [state[3] for state in states]
        phi_dg = [state[4] for state in states]
        psi_dg = [state[5] for state in states]
        time_vector = [state[10] for state in states]

        #return a dataframe
        planner_states = pd.DataFrame({'x':x, 'y':y, 'z':z, 
                                    'theta_dg':theta_dg, 
                                    'phi_dg':phi_dg, 
                                    'psi_dg':psi_dg, 
                                    'time':time_vector})

        return planner_states

    
    @staticmethod
    def scale_cartesian_with_terrain(cartesian_states:pd.DataFrame,
        terrain:Terrain) -> pd.DataFrame:
        
        """
        This function takes in a dataframe of planner states and formats
        the x and y positions to be in the same coordinate system as the
        terrain map
        input: planner_states dataframe
        output: formatted dataframe
        
        """
        x_formatted = []
        y_formatted = []
        z_formatted = []
        
        for i in range(len(cartesian_states)):
            x = cartesian_states['x'][i]
            y = cartesian_states['y'][i]
            
            x_pos = x/(terrain.max_x - terrain.min_x) * terrain.expanded_array.shape[0]
            y_pos = y/(terrain.max_y - terrain.min_y) * terrain.expanded_array.shape[1]
            
            x_formatted.append(x_pos)
            y_formatted.append(y_pos)
            z_formatted.append(cartesian_states['z'][i])
            
        formatted_df = pd.DataFrame({'x':x_formatted, 'y':y_formatted,
                                        'z':z_formatted})
        
        return formatted_df 
    
    #can probably optimize with jit
    @staticmethod
    def format_radar_data_with_terrain(radar_voxels:dict, terrain:Terrain) -> dict:
        """
        This function takes in a dictionary of radar voxels and formats
        the x and y positions to be in the same coordinate system as the
        terrain map
        input: radar_voxels dictionary
        output: formatted dictionary
        
        """
        x_formatted = []
        y_formatted = []
        z_formatted = []
        
        for i in range(len(radar_voxels['voxel_x'])):
            x = radar_voxels['voxel_x'][i]
            y = radar_voxels['voxel_y'][i]
            
            x_pos = x/(terrain.max_x - terrain.min_x) * terrain.expanded_array.shape[0]
            y_pos = y/(terrain.max_y - terrain.min_y) * terrain.expanded_array.shape[1]
            
            x_formatted.append(x_pos)
            y_formatted.append(y_pos)
            z_formatted.append(radar_voxels['voxel_z'][i])
            
        formatted_dict = {'voxel_x':x_formatted, 'voxel_y':y_formatted,
                                        'voxel_z':z_formatted,
                                        'voxel_vals':radar_voxels['voxel_vals']}
        
        return formatted_dict