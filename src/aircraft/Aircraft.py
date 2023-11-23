"""

"""
import numpy as np 

class AircraftInfo():
    """
    Represents an aircraft with its parameters, initial states, and controls.

    Parameters:
        aircraft_params (dict): A dictionary containing aircraft parameters.
        aircraft_init_states (dict): A dictionary containing initial states of the aircraft.
        aircraft_init_controls (dict): A dictionary containing initial control inputs for the aircraft.

    Attributes:
        x, y, z (float): Inertial position coordinates in NED (North-East-Down) frame.
        u, v, w (float): Body velocity components in aircraft's body frame.
        phi, theta, psi (float): Attitude angles representing roll, pitch, and yaw in radians.
        p, q, r (float): Body angular velocities around the roll, pitch, and yaw axes in rad/s.
        delta_e, delta_a, delta_r, delta_t (float): Control inputs representing elevator, aileron, rudder, and throttle.
        states (numpy.array): Array containing current state variables [x, y, z, u, v, w, phi, theta, psi, p, q, r].
        controls (numpy.array): Array containing current control inputs [delta_e, delta_a, delta_r, delta_t].

    Methods:
        init_states(init_states: dict) -> None:
            Initializes the aircraft states with the provided initial states.
            
        init_controls(controls: dict) -> None:
            Initializes the aircraft controls with the provided initial control inputs.
            
        update_positions(ned_x: float, ned_y: float, ned_z: float) -> None:
            Updates the inertial position of the aircraft.
            
        update_velocities(u: float, v: float, w: float) -> None:
            Updates the body velocity of the aircraft.
            
        update_attitudes(phi_rad: float, theta_rad: float, psi_rad: float) -> None:
            Updates the inertial attitudes of the aircraft in radians.
            
        update_angular_velocities(p: float, q: float, r: float) -> None:
            Updates the body angular velocities of the aircraft in rad/s.
            
        update_controls(delta_e_rad: float, delta_a_rad: float, 
                        delta_r_rad: float, delta_t_newtons: float) -> None:
            Updates the controls of the aircraft. Inputs for delta_e, delta_a, delta_r must be in radians, 
            and delta_t in Newtons.
    """
    def __init__(self, aircraft_params:dict, 
                 aircraft_init_states:dict, 
                 aircraft_init_controls:dict) -> None:
        self.aircraft_params = aircraft_params
        
        self.x = None
        self.y = None
        self.z = None
        self.u = None
        self.v = None
        self.w = None
        self.phi = None
        self.theta = None
        self.psi = None
        self.p = None
        self.q = None
        self.r = None

        self.delta_e = None
        self.delta_a = None
        self.delta_r = None
        self.delta_t = None
        
        self.states = self.init_states(aircraft_init_states)
        # self.controls = self.init_controls(aircraft_init_controls)
        self.rho = 1.225 #kg/m^3


    def init_states(self, init_states:dict) -> None:
        """
        This method initializes the states of the aircraft 
        with the initial states provided in the dictionary
        """        

        #check if we have the correct keys
        for key in init_states.keys():
            if key not in ['x', 'y', 'z', 
                           'u', 'v', 'w', 
                           'phi', 'theta', 
                           'psi', 
                           'p', 'q', 'r', 'h']:
                raise ValueError("The key {} is not valid".format(key))
        
        #position frame are in inertial ned
        self.x = init_states['x']
        self.y = init_states['y']
        
        #the dictionary might be using h instead of z
        if 'z' not in init_states.keys():
            self.z = init_states['h']
            self.use_z_key = False
        else:
            self.z = init_states['z']   
            self.use_z_key = True

        #velocity frame are in body
        self.u = init_states['u']
        self.v = init_states['v']
        self.w = init_states['w']

        #attitude frame are in body
        self.phi = init_states['phi']
        self.theta = init_states['theta']
        self.psi = init_states['psi']

        #angular velocity frame are in body
        self.p = init_states['p']
        self.q = init_states['q']
        self.r = init_states['r']

        self.states = np.array([self.x, self.y, self.z, 
                                self.u, self.v, self.w, 
                                self.phi, self.theta, self.psi,
                                self.p, self.q, self.r])
        
        return self.states
        
    def init_controls(self, controls:dict) -> None:
        """
        This method initializes the controls of the aircraft 
        with the initial controls provided in the dictionary
        """        

        #check if we have the correct keys
        for key in controls.keys():
            if key not in ['delta_e', 'delta_a', 'delta_r', 'delta_t']:
                raise ValueError("The key {} is not valid".format(key))
            return
        
        #position frame are in inertial ned
        self.delta_e = float(controls['delta_e'])
        self.delta_a = float(controls['delta_a'])
        self.delta_r = float(controls['delta_r'])
        self.delta_t = float(controls['delta_t'])

        self.controls = np.array([self.delta_e, self.delta_a, 
                                  self.delta_r, self.delta_t], dtype=float)
    
        return self.controls

    def update_positions(
            self, ned_x:float, ned_y:float, ned_z:float) -> None:
        """
        This method updates the inertial position of the aircraft
        """
        self.x = float(ned_x)
        self.y = float(ned_y)
        self.z = float(ned_z)
        self.states[0:3] = np.array([self.x, self.y, self.z], dtype=float)

    def update_velocities(
            self, u:float, v:float, w:float) -> None:
        """
        This method updates the body velocity of the aircraft
        """
        self.u = float(u)
        self.v = float(v)
        self.w = float(w)
        self.states[3:6] = np.array([self.u, self.v, self.w], dtype=float)

    def update_attitudes(
            self, phi_rad:float, theta_rad:float, psi_rad:float) -> None:
        """
        This method updates the inertial attitudes of the aircraft 
        inputs must be in radians 
        """
        self.phi = phi_rad
        self.theta = theta_rad
        self.psi = psi_rad
        self.states[6:9] = np.array([self.phi, self.theta, self.psi], dtype=float)

    def update_angular_velocities(
            self, p:float, q:float, r:float) -> None:
        """
        This method updates the body angular velocities of the aircraft
        inputs must be in rad/s 
        """
        self.p = p
        self.q = q
        self.r = r
        self.states[9:12] = np.array([self.p, self.q, self.r], dtype=float)


    def update_controls(
            self, delta_e_rad:float, delta_a_rad:float, 
            delta_r_rad:float, delta_t_newtons:float) -> None:
        """
        This method updates the controls of the aircraft inputs for 
        delta_e, delta_a, delta_r must be in radians
        delta_t in Newtons
        """
        self.delta_e = float(delta_e_rad)
        self.delta_a = float(delta_a_rad)
        self.delta_r = float(delta_r_rad)
        self.delta_t = float(delta_t_newtons)
        self.controls = np.array([self.delta_e, self.delta_a, 
                                  self.delta_r, self.delta_t], dtype=float)
        
    def update_states(self, new_states:np.ndarray) -> None:
        """
        Updates the states of the aircraft with the provided array
        """
        self.update_positions(new_states[0], new_states[1], new_states[2])
        self.update_velocities(new_states[3], new_states[4], new_states[5])
        self.update_attitudes(new_states[6], new_states[7], new_states[8])
        self.update_angular_velocities(new_states[9], new_states[10], new_states[11])

    def get_states(self) -> np.ndarray:
        """
        Returns the states of the aircraft
        """
        return self.states
    
    def get_controls(self) -> np.ndarray:
        """
        Returns the controls of the aircraft
        """
        return self.controls
    