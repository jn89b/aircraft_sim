import casadi as ca
import numpy as np
from src.mpc.MPC import ModelPredictiveControl

class LinearizedAircraftMPC(ModelPredictiveControl):
    def __init__(self, mpc_params:dict, 
                 airplane_constraint_params:dict):
        super().__init__(mpc_params)
        self.airplane_params = airplane_constraint_params
        
    def addAdditionalConstraints(self):
        self.lbx['U'][0, :] = self.airplane_params['delta_e_min']
        self.ubx['U'][0, :] = self.airplane_params['delta_e_max']
        
        self.lbx['U'][1, :] = self.airplane_params['delta_t_min']
        self.ubx['U'][1, :] = self.airplane_params['delta_t_max']
        
        self.lbx['U'][2, :] = self.airplane_params['delta_a_min']
        self.ubx['U'][2, :] = self.airplane_params['delta_a_max']
        
        self.lbx['U'][3, :] = self.airplane_params['delta_r_min']
        self.ubx['U'][3, :] = self.airplane_params['delta_r_max']
        
        self.lbx['X'][0, :] = self.airplane_params['u_min']
        self.ubx['X'][0, :] = self.airplane_params['u_max']
        
        # self.lbx['X'][1, :] = self.airplane_params['w_min']
        # self.ubx['X'][1, :] = self.airplane_params['w_max']
        
        self.lbx['X'][2, :] = self.airplane_params['q_min']
        self.ubx['X'][2, :] = self.airplane_params['q_max']
        
        self.lbx['X'][3, :] = self.airplane_params['theta_min']
        self.ubx['X'][3, :] = self.airplane_params['theta_max']
        
        # self.lbx['X'][4, :] = self.airplane_params['h_min']
        # self.ubx['X'][4, :] = self.airplane_params['h_max']
        
        # self.lbx['X'][5, :] = self.airplane_params['x_min']
        # self.ubx['X'][5, :] = self.airplane_params['x_max']
    
        self.lbx['X'][6, :] = self.airplane_params['v_min']
        self.ubx['X'][6, :] = self.airplane_params['v_max']
        
        self.lbx['X'][7, :] = self.airplane_params['p_min']
        self.ubx['X'][7, :] = self.airplane_params['p_max']
        
        # self.lbx['X'][8, :] = self.airplane_params['r_min']
        # self.ubx['X'][8, :] = self.airplane_params['r_max']
        
        self.lbx['X'][9, :] = self.airplane_params['phi_min']
        self.ubx['X'][9, :] = self.airplane_params['phi_max']
        
        # self.lbx['X'][10, :] = self.airplane_params['psi_min']
        # self.ubx['X'][10, :] = self.airplane_params['psi_max']
        
        # self.lbx['X'][11, :] = self.airplane_params['y_min']
        # self.ubx['X'][11, :] = self.airplane_params['y_max']
        
    def unpack_controls(self, u:ca.DM) -> dict:
        """
        unpack the control variables
        """
        control_dict = {
            'delta_e': u[0,:],
            'delta_t': u[1,:],
            'delta_a': u[2,:],
            'delta_r': u[3,:]
        }

        for k,v in control_dict.items():
            control_dict[k] = np.array(v).reshape(-1)

        return control_dict
    
    def unpack_states(self, x:ca.DM) -> dict:
        """
        unpack the state variables
        """
        #reshape to -1
        # x = np.array(x).reshape(-1, 12)
        state_dict = {
            'u': np.array(x[0,:]),
            'w': np.array(x[1,:]),
            'q': np.array(x[2,:]),
            'theta': np.array(x[3,:]),
            'h': np.array(x[4,:]),
            'x': np.array(x[5,:]),
            'v': np.array(x[6,:]),
            'p': np.array(x[7,:]),
            'r': np.array(x[8,:]),
            'phi': np.array(x[9,:]),
            'psi': np.array(x[10,:]),
            'y': np.array(x[11,:])}

        for k,v in state_dict.items():
            state_dict[k] = np.array(v).reshape(-1)
        
        return state_dict
    
class LongitudinalMPC(ModelPredictiveControl):
    def __init__(self, mpc_params:dict, 
                 airplane_constraint_params:dict):
        super().__init__(mpc_params)
        self.airplane_params = airplane_constraint_params
        
    def addAdditionalConstraints(self):
        self.lbx['U'][0, :] = self.airplane_params['delta_e_min']
        self.ubx['U'][0, :] = self.airplane_params['delta_e_max']
        
        self.lbx['U'][1, :] = self.airplane_params['delta_t_min']
        self.ubx['U'][1, :] = self.airplane_params['delta_t_max']

        self.lbx['X'][0, :] = self.airplane_params['u_min']
        self.ubx['X'][0, :] = self.airplane_params['u_max']
        
        # self.lbx['X'][1, :] = self.airplane_params['w_min']
        # self.ubx['X'][1, :] = self.airplane_params['w_max']
        
        self.lbx['X'][2, :] = self.airplane_params['q_min']
        self.ubx['X'][2, :] = self.airplane_params['q_max']
        
        self.lbx['X'][3, :] = self.airplane_params['theta_min']
        self.ubx['X'][3, :] = self.airplane_params['theta_max']
        
        # self.lbx['X'][4, :] = self.airplane_params['h_min']
        # self.ubx['X'][4, :] = self.airplane_params['h_max']
        
        # self.lbx['X'][5, :] = self.airplane_params['x_min']
        # self.ubx['X'][5, :] = self.airplane_params['x_max']
                
        
    def unpack_controls(self, u:ca.DM) -> dict:
        """
        unpack the control variables
        """
        control_dict = {
            'delta_e': u[0,:],
            'delta_t': u[1,:]
        }

        for k,v in control_dict.items():
            control_dict[k] = np.array(v).reshape(-1)

        return control_dict
    
    def unpack_states(self, x:ca.DM) -> dict:
        """
        unpack the state variables
        """
        #reshape to -1
        # x = np.array(x).reshape(-1, 12)
        state_dict = {
            'u': np.array(x[0,:]),
            'w': np.array(x[1,:]),
            'q': np.array(x[2,:]),
            'theta': np.array(x[3,:]),
            'h': np.array(x[4,:]),
            'x': np.array(x[5,:])}

        for k,v in state_dict.items():
            state_dict[k] = np.array(v).reshape(-1)
        
        return state_dict
        
class LateralMPC(ModelPredictiveControl):
    def __init__(self, mpc_params:dict, 
                 airplane_constraint_params:dict):
        super().__init__(mpc_params)
        self.airplane_params = airplane_constraint_params
        
    def addAdditionalConstraints(self):
        self.lbx['U'][0, :] = self.airplane_params['delta_a_min']
        self.ubx['U'][0, :] = self.airplane_params['delta_a_max']
        
        self.lbx['U'][1, :] = self.airplane_params['delta_r_min']
        self.ubx['U'][1, :] = self.airplane_params['delta_r_max']
        
        # self.lbx['X'][0, :] = self.airplane_params['v_min']
        # self.ubx['X'][0, :] = self.airplane_params['v_max']
        
        self.lbx['X'][1, :] = self.airplane_params['p_min']
        self.ubx['X'][1, :] = self.airplane_params['p_max']
        
        self.lbx['X'][2, :] = self.airplane_params['r_min']
        self.ubx['X'][2, :] = self.airplane_params['r_max']
        
        self.lbx['X'][3, :] = self.airplane_params['phi_min']
        self.ubx['X'][3, :] = self.airplane_params['phi_max']
        
        # self.lbx['X'][4, :] = self.airplane_params['psi_min']
        # self.ubx['X'][4, :] = self.airplane_params['psi_max']
        
        # self.lbx['X'][5, :] = self.airplane_params['y_min']
        # self.ubx['X'][5, :] = self.airplane_params['y_max']
        
    def unpack_controls(self, u:ca.DM) -> dict:
        """
        unpack the control variables
        """
        control_dict = {
            'delta_a': u[0,:],
            'delta_r': u[1,:]
        }

        for k,v in control_dict.items():
            control_dict[k] = np.array(v).reshape(-1)

        return control_dict
    
    def unpack_states(self, x:ca.DM) -> dict:
        """
        unpack the state variables
        """
        #reshape to -1
        # x = np.array(x).reshape(-1, 12)
        state_dict = {
            'v': np.array(x[0,:]),
            'p': np.array(x[1,:]),
            'r': np.array(x[2,:]),
            'phi': np.array(x[3,:]),
            'psi': np.array(x[4,:]),
            'y': np.array(x[5,:])}

        for k,v in state_dict.items():
            state_dict[k] = np.array(v).reshape(-1)
        
        return state_dict
        
class FixedWingMPC(ModelPredictiveControl):
    def __init__(self, mpc_params:dict, 
                 airplane_constraint_params:dict):
        super().__init__(mpc_params)
        self.airplane_params = airplane_constraint_params
        # self.S = 0.5

    def addAdditionalConstraints(self):
        self.lbx['U'][0, :] = self.airplane_params['delta_a_min']
        self.ubx['U'][0, :] = self.airplane_params['delta_a_max']

        self.lbx['U'][1, :] = self.airplane_params['delta_e_min']
        self.ubx['U'][1, :] = self.airplane_params['delta_e_max']

        self.lbx['U'][2, :] = self.airplane_params['delta_r_min']
        self.ubx['U'][2, :] = self.airplane_params['delta_r_max']

        self.lbx['U'][3, :] = self.airplane_params['delta_t_min']
        self.ubx['U'][3, :] = self.airplane_params['delta_t_max']

        self.lbx['X'][2, :] = self.airplane_params['z_min']
        self.ubx['X'][2, :] = self.airplane_params['z_max']

        self.lbx['X'][3, :] = self.airplane_params['u_min']
        self.ubx['X'][3, :] = self.airplane_params['u_max']

        # self.lbx['X'][4, :] = self.airplane_params['v_min']
        # self.ubx['X'][4, :] = self.airplane_params['v_max']

        # self.lbx['X'][5, :] = self.airplane_params['w_min']
        # self.ubx['X'][5, :] = self.airplane_params['w_max']

        self.lbx['X'][6, :] = self.airplane_params['phi_min']
        self.ubx['X'][6, :] = self.airplane_params['phi_max']

        self.lbx['X'][7, :] = self.airplane_params['theta_min']
        self.ubx['X'][7, :] = self.airplane_params['theta_max']

        # self.lbx['X'][8, :] = self.airplane_params['psi_min']
        # self.ubx['X'][8, :] = self.airplane_params['psi_max']

        self.lbx['X'][9, :] = self.airplane_params['p_min']
        self.ubx['X'][9, :] = self.airplane_params['p_max']

        self.lbx['X'][10, :] = self.airplane_params['q_min']
        self.ubx['X'][10, :] = self.airplane_params['q_max']

        self.lbx['X'][11, :] = self.airplane_params['r_min']
        self.ubx['X'][11, :] = self.airplane_params['r_max']

    def unpack_controls(self, u:ca.DM) -> dict:
        """
        unpack the control variables
        """
        control_dict = {
            'delta_a': u[0,:],
            'delta_e': u[1,:],
            'delta_r': u[2,:],
            'delta_t': u[3,:]
        }

        for k,v in control_dict.items():
            control_dict[k] = np.array(v).reshape(-1)

        return control_dict
    
    def unpack_states(self, x:ca.DM) -> dict:
        """
        unpack the state variables
        """
        #reshape to -1
        # x = np.array(x).reshape(-1, 12)
        state_dict = {
            'x': np.array(x[0,:]),
            'y': np.array(x[1,:]),
            'z': np.array(x[2,:]),
            'u': np.array(x[3,:]),
            'v': np.array(x[4,:]),
            'w': np.array(x[5,:]),
            'phi': np.array(x[6,:]),
            'theta': np.array(x[7,:]),
            'psi': np.array(x[8,:]),
            'p': np.array(x[9,:]),
            'q': np.array(x[10,:]),
            'r': np.array(x[11,:])
        }

        for k,v in state_dict.items():
            state_dict[k] = np.array(v).reshape(-1)
        

        return state_dict




