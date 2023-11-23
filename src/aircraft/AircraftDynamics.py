import numpy as np
import math
import random
import casadi as ca

from src.aircraft.Aircraft import AircraftInfo
from src.config import Config

from src.math_lib.VectorOperations import euler_dcm_inertial_to_body, \
    compute_B_matrix, euler_dcm_body_to_inertial, ca_compute_B_matrix    
    
from src.math_lib.VectorOperations import ca_euler_dcm_body_to_inertial, \
    ca_euler_dcm_inertial_to_body
 

def convert_lin_states_to_regular(states:np.ndarray) -> np.ndarray:
    """
     Given states modeled as linearized states :
        [u, 
         w, 
         q, 
         theta, 
         h, 
         x, 
         v, 
         p, 
         r, 
         phi, 
         psi, 
         y]
        
    Returns the states in the regular order:
        [x, 
         y, 
         z, 
         u, 
         v, 
         w, 
         phi, 
         theta, 
         psi, 
         p, 
         q, 
         r]
        
    To do:
        - Have this use an index configuration to avoid confusion
    """
    return np.array([states[5], 
                     states[11], 
                     states[4], 
                     states[0], 
                     states[6], 
                     states[1], 
                     states[9], 
                     states[3], 
                     states[10], 
                     states[7], 
                     states[2], 
                     states[8]])
    
def convert_lin_controls_to_regular(controls:np.ndarray) -> np.ndarray:
    """
    Given controls modeled as linearized controls:
        [delta_e, delta_t, delta_a, delta_r]
        
    Returns the controls in the regular order:
        [delta_a, delta_e, delta_r, delta_t]
        
    To do:
        - Have this use an index configuration to avoid confusion
    """ 
    return np.array([controls[2],
                    controls[0],
                    controls[3],
                    controls[1]])
    
    
def convert_regular_states_to_lin(states:np.ndarray) -> np.ndarray:
    """
    Given states in the regular order:
        [x, y, z, u, v, w, phi, theta, psi, p, q, r]
        
    Returns the states modeled as linearized states :
        [u, w, q, theta, h, x, v, p, r, phi, psi, y]
        
    To do:
        - Have this use an index configuration to avoid confusion
    """
    return np.array([states[3], 
                     states[1], 
                     states[10], 
                     states[7], 
                     states[2], 
                     states[0], 
                     states[6], 
                     states[11], 
                     states[9], 
                     states[5], 
                     states[4], 
                     states[8]])
    
def convert_regular_controls_to_lin(controls:np.ndarray) -> np.ndarray:
    """
    Given controls in the regular order:
        [delta_a, delta_e, delta_r, delta_t]
        
    Returns the controls modeled as linearized controls orientation:
        [delta_e, delta_t, delta_a, delta_r]
        
    To do:
        - Have this use an index configuration to avoid confusion
    """ 
    return np.array([controls[1],
                    controls[3],
                    controls[0],
                    controls[2]])
    
class AircraftDynamicsV2():
    def __init__(self, aircraft:AircraftInfo) -> None:
        self.aircraft = aircraft
        self.thrust_scale = self.aircraft.aircraft_params['mass'] * 9.81 / \
            Config.HOVER_THROTTLE
            
    
    def compute_alpha(self, u:float, w:float) -> float:
        """
        Computes the angle of attack
        """
        #compute the angle of attack
        return np.arctan2(w, u)        
    
    def compute_beta(self, v:float, airspeed:float) -> float:
        """
        Computes the sideslip angle
        """
        #check divide by zero 
        # if u == 0:
        #     return 0.
        beta_rad = np.arcsin(v/airspeed)
        return beta_rad

    def lift_coeff(self, alpha:float) -> float:
        """
        Computes the lift coefficient with a linear and flat plate model
        """
        coefficient = self.aircraft.aircraft_params
        alpha0 = coefficient["alpha_stall"]
        M = coefficient["mcoeff"]
        c_lift_0 = coefficient["c_lift_0"]
        c_lift_a0 = coefficient["c_lift_a"]

        max_alpha_delta = 0.8
        if alpha - alpha0 > max_alpha_delta:
            alpha = alpha0 + max_alpha_delta
        elif alpha0 - alpha > max_alpha_delta:
            alpha = alpha0 - max_alpha_delta

        sigmoid = (1 + np.exp(-M * (alpha - alpha0)) + \
                   np.exp(M * (alpha + alpha0))) / (1 + math.exp(-M * (alpha - alpha0))) \
                    / (1 + math.exp(M * (alpha + alpha0)))
        linear = (1.0 - sigmoid) * (c_lift_0 + c_lift_a0 * alpha)  # Lift at small AoA
        flat_plate = sigmoid * (2 * math.copysign(1, alpha) * math.pow(math.sin(alpha), 2) * math.cos(alpha))  # Lift beyond stall

        return linear + flat_plate
    
    def drag_coeff(self, alpha_rad) -> float:
        coefficient = self.aircraft.aircraft_params
        b = coefficient["b"]
        s = coefficient["s"]
        c_drag_p = coefficient["c_drag_p"]
        c_lift_0 = coefficient["c_lift_0"]
        c_lift_a0 = coefficient["c_lift_a"]
        oswald = coefficient["oswald"]
        
        ar = pow(b, 2) / s
        c_drag_a = c_drag_p + pow(c_lift_0 + c_lift_a0 * alpha_rad, 2) / \
            (np.pi * oswald * ar)

        return c_drag_a
    
    
    def compute_cy(self, beta_rad:float,
                   states:np.ndarray, 
                   aileron_input_rad:float,
                   rudder_input_rad:float) -> float:
        """
        x y z u v w phi theta psi p q r
        """
        coefficients = self.aircraft.aircraft_params
        c_y_b = coefficients["c_y_b"]
        c_y_p = coefficients["c_y_p"]
        c_y_r = coefficients["c_y_r"]
        c_y_deltaa = coefficients["c_y_deltaa"]
        c_y_deltar = coefficients["c_y_deltar"]
        
        # CYbeta  =	interp1(AlphaTable,CYBetaTable,alphadeg);
        # CYp     =   interp1(AlphaTable,CYpHatTable,alphadeg)*b/(2*V);
        # CYr     =   interp1(AlphaTable,CYrHatTable,alphadeg)*b/(2*V);
        # CYdA    =   interp1(AlphaTable,CYdATable,alphadeg);	
        # CYdR    =   interp1(AlphaTable,CYdRTable,alphadeg);	
        # CYdAS   =   interp1(AlphaTable,CYdASTable,alphadeg);

        # CY      =   CYbeta*betar + CYp*x(7) + CYr*x(9) ...
        #             + CYdA*u(2) + CYdR*u(3) + CYdAS*u(5);
        
        
        CY = (c_y_b * beta_rad) + (c_y_p * states[9]) + \
            (c_y_r * states[11]) + (c_y_deltaa * aileron_input_rad) + \
                (c_y_deltar * rudder_input_rad)
            
        return CY


    def compute_Cm(self, alpha_rad:float, 
                   input_elevator_rad:float) -> float:
        """
        Computes the pitching moment coefficient
        """
        coefficient = self.aircraft.aircraft_params
        c_m_a = coefficient["c_m_a"]
        c_m_q = coefficient["c_m_q"]
        c_m_deltae = coefficient["c_m_deltae"]

        # CmStat  =	interp1(AlphaTable,CmTable,alphadeg);
        # Cmq     =   interp1(AlphaTable,CmqHatTable,alphadeg)*cBar/(2*V);
        # CmdE	=	interp1(AlphaTable,CmdETable,alphadeg);
        # CmdF	=	interp1(AlphaTable,CmdFTable,alphadeg);
        # CmdS	=	interp1(AlphaTable,CmdSTable,alphadeg);

        # Cm  =	CmStat - CL*SMsim + Cmq*x(8) + CmdE*u(1)+ CmdF*u(6) + CmdS*u(7);

        #Cm = c_m_0 + c_m_a * alpha_rad + c_m_q * alpha_rad + c_m_deltae
        Cm = (c_m_a * alpha_rad) + (c_m_q * alpha_rad) + (c_m_deltae*input_elevator_rad)
        return Cm
    
    def compute_roll_momemt_coefficient(self, beta_rad:float,
                                        states:np.ndarray,
                                        input_aileron_rad:float,
                                        input_rudder_rad:float) -> float:
	
        # Clbeta	=	interp1(AlphaTable,ClBetaTable,alphadeg);
        # Clp	    =	interp1(AlphaTable,ClpHatTable,alphadeg)*b/(2*V);				
        # Clr	    =	interp1(AlphaTable,ClrHatTable,alphadeg)*b/(2*V);				
        # CldA    =	interp1(AlphaTable,CldATable,alphadeg);
        # CldR	=	interp1(AlphaTable,CldRTable,alphadeg);
        # CldAS   =   interp1(AlphaTable,CYdASTable,alphadeg);
        
        
        # Cl      =	Clbeta*betar + Clr*x(9) + Clp*x(7) ...
        #             + (CldA*u(2) + CldR*u(3) + CldAS*u(5));
        # x9 is r 
        # x7 is p
        # u2 is aileron
        # u3 is rudder
        aircraft_params = self.aircraft.aircraft_params
        c_l_b = aircraft_params["c_l_b"]
        c_l_p = aircraft_params["c_l_p"]
        c_l_r = aircraft_params["c_l_r"]
        c_l_deltaa = aircraft_params["c_l_deltaa"]
        c_l_deltar = aircraft_params["c_l_deltar"]
        
        r = states[11]
        p = states[9]
        
        
        roll_moment = (c_l_b * beta_rad) + (c_l_p * p) + \
            (c_l_r * r) + (c_l_deltaa * input_aileron_rad) + \
                (c_l_deltar * input_rudder_rad)
        
        return roll_moment
    
    def compute_Cn(self, beta_rad:float,
                   states:np.ndarray, input_aileron_rad:float,
                   input_rudder_rad:float) -> float:

        """
        x y z u v w phi theta psi p q r
        """
        # Cnbeta	=	interp1(AlphaTable,CnBetaTable,alphadeg);
        # Cnp	    =	interp1(AlphaTable,CnpHatTable,alphadeg)*b/(2*V);				
        # Cnr	    =	interp1(AlphaTable,CnrHatTable,alphadeg)*b/(2*V);				
        # CndA    =	interp1(AlphaTable,CndATable,alphadeg);
        # CndR	=	interp1(AlphaTable,CndRTable,alphadeg);
        # CndAS   =   interp1(AlphaTable,CndASTable,alphadeg);

        # Cn      =	Cnbeta*betar + Cnp*x(7) + Cnr*x(9) ...
        #             + (CndA*u(2) + CndR*u(3) + CndAS*u(5));
        
        coefficient = self.aircraft.aircraft_params
        c_n_0 = coefficient["c_n_0"]
        c_n_b = coefficient["c_n_b"]
        c_n_p = coefficient["c_n_p"]
        c_n_r = coefficient["c_n_r"]
        c_n_deltaa = coefficient["c_n_deltaa"]
        c_n_deltar = coefficient["c_n_deltar"]
        
        Cn = (c_n_b * beta_rad) + (c_n_p * states[9]) + \
            (c_n_r * states[7]) + (c_n_deltaa * input_aileron_rad) + \
                (c_n_deltar * input_rudder_rad)
        
        return Cn

    
    def compute_derivatives(self, input_aileron_rad:float,
                            input_elevator_rad:float,
                            input_rudder_rad:float,
                            input_thrust_percent:float, 
                            current_states:np.ndarray) -> np.ndarray:
        """
        Computes the derivatives of the aircraft 
        """
        # unpacking states
        b = self.aircraft.aircraft_params['b']
        c_l_deltae = self.aircraft.aircraft_params['c_lift_deltae']
        c = self.aircraft.aircraft_params['c']
        current_states = self.aircraft.states
        u = current_states[3]
        v = current_states[4]
        w = current_states[5]
        phi = current_states[6]
        theta = current_states[7]
        psi = current_states[8]
        p = current_states[9]
        q = current_states[10]
        r = current_states[11]
        
        #check theta     
        airspeed = np.sqrt(u**2 + v**2 + w**2)
        
        alpha_rad = self.compute_alpha(u, 
                                       v)
        
        beta_rad = self.compute_beta(v,
                                     airspeed)

        CL = self.lift_coeff(alpha_rad) + (c_l_deltae*input_elevator_rad)             
        CD = self.drag_coeff(alpha_rad)
        
        CX = -CD*np.cos(alpha_rad) + CL*np.sin(alpha_rad)
        CY = self.compute_cy(beta_rad,
                             current_states, 
                             input_aileron_rad,
                             input_rudder_rad)
        CZ = -CD*np.sin(alpha_rad) - CL*np.cos(alpha_rad)
        
        CM = self.compute_Cm(alpha_rad, input_elevator_rad)
        CN = self.compute_Cn(beta_rad,
                             current_states,
                             input_aileron_rad,
                             input_rudder_rad)
        roll_moment = self.compute_roll_momemt_coefficient(beta_rad,
            current_states, input_aileron_rad, input_rudder_rad)
                                                              
        
        Q = 0.5 * self.aircraft.rho * airspeed**2
        S = self.aircraft.aircraft_params['s']
        m = self.aircraft.aircraft_params['mass']
        # map thrust to newtons
        input_thrust_n = input_thrust_percent * self.thrust_scale
        
        #Earth to Body DCM
        phi = current_states[6]
        theta = current_states[7]
        psi = current_states[8]
        
        HEB = euler_dcm_inertial_to_body(phi, theta, psi)
        gravity_earth = np.array([0, 0, Config.G])
        gravity_body = np.matmul(HEB, gravity_earth)
         
        # state accelerations
        Xb = ((Q*S*CX) + input_thrust_n) / m
        Yb = (Q*S*CY) / m
        Zb = (Q*S*CZ) / m
        
        # state moments
        Lb = roll_moment*Q*S*b
        Mb = (Q*S*b*CM*c) / m
        Nb = (Q*S*b*CN) / m
        
        #include CG offset
        CGOffset_x = self.aircraft.aircraft_params["CGOffset_x"]
        CGOffset_y = self.aircraft.aircraft_params["CGOffset_y"]
        CGOffset_z = self.aircraft.aircraft_params["CGOffset_z"]
        
        Lb +=  CGOffset_y*Zb - CGOffset_z*Yb
        Mb += -CGOffset_x*Zb + CGOffset_z*Xb
        Nb += -CGOffset_y*Xb + CGOffset_x*Yb        
        
        nz = -Zb/Config.G #normal load factor
        
        # dynamic equations
        # accelerations in the body frame
        u_dot = Xb + gravity_body[0] + (r*v) - (q*w) 
        v_dot = Yb + gravity_body[1] - (r*u) + (p*w)
        w_dot = Zb + gravity_body[2] + (q*u) - (p*v)
        
        #
        rotation = np.matmul(HEB, np.array([u, v, w]))
        x_dot = rotation[0]
        y_dot = rotation[1]
        z_dot = rotation[2]
        
        #moments
        Ixx = self.aircraft.aircraft_params["Ixx"]
        Iyy = self.aircraft.aircraft_params["Iyy"]
        Izz = self.aircraft.aircraft_params["Izz"]
        Ixz = 0.0
  
        p_dot = (Lb +  (Ixx - Iyy) * q * r)  / Ixx
        q_dot = (Mb - (Izz - Iyy) * p * q) / Iyy
        r_dot = (Nb + (Ixx - Izz) * p * q) / Izz
        # p_dot	= Izz*Lb -(Izz*(Izz - Iyy)*r)*q / (Ixx*Izz)
        # q_dot   = Mb - (Ixx - Izz)*p*r / Iyy
        # r_dot   = Ixx*Nb - (Ixx*(Ixx - Iyy)*p)*q / (Ixx*Izz)
        
        B = compute_B_matrix(phi, theta, psi)
        phi_theta_psi_dot = np.matmul(B, np.array([p, q, r]))
        
        phi_dot = phi_theta_psi_dot[0]
        theta_dot = phi_theta_psi_dot[1]
        psi_dot = phi_theta_psi_dot[2]
                
        derivatives = np.array([x_dot, y_dot, z_dot,
                                u_dot, v_dot, w_dot,
                                phi_dot, theta_dot, psi_dot,
                                p_dot, q_dot, r_dot])
        
        return derivatives

    
    def rk45(self, 
             input_aileron_rad:float,
             input_elevator_rad:float,
             input_rudder_rad:float,
             input_thrust_n:float,
             states:np.ndarray,
             delta_time:float) -> np.ndarray:
        """
        Simulates the aircraft using the Runge-Kutta 4th order method 
        """
        #get the current states
        current_states = states

        #compute the derivatives
        k1 = delta_time * self.compute_derivatives(input_aileron_rad,
                                        input_elevator_rad,
                                        input_rudder_rad,
                                        input_thrust_n,
                                        current_states)
        
        k2 = delta_time * self.compute_derivatives(input_aileron_rad,
                                        input_elevator_rad,
                                        input_rudder_rad,
                                        input_thrust_n,
                                        current_states + k1/2)
        
        k3 = delta_time * self.compute_derivatives(input_aileron_rad,
                                        input_elevator_rad,
                                        input_rudder_rad,
                                        input_thrust_n,
                                        current_states + k2/2)
        
        k4 = delta_time * self.compute_derivatives(input_aileron_rad,
                                        input_elevator_rad,
                                        input_rudder_rad,
                                        input_thrust_n,
                                        current_states + k3)
        
        new_states = current_states + (k1 + 2*k2 + 2*k3 + k4) / 6
     

        return new_states
        
    def eulers(self, 
               input_aileron_rad:float,
               input_elevator_rad:float,
               input_rudder_rad:float,
               input_thrust_n:float,
               states:np.ndarray,
               delta_time:float) -> np.ndarray:
        """
        Simulates the aircraft using the Euler's method 
        """
        #get the current states
        current_states = states

        #compute the derivatives
        derivatives = self.compute_derivatives(input_aileron_rad,
                                        input_elevator_rad,
                                        input_rudder_rad,
                                        input_thrust_n,
                                        current_states)
        
        new_states = current_states + derivatives * delta_time
     

        return new_states
    
class AircraftDynamics():
    def __init__(self, aircraft:AircraftInfo) -> None:
        self.aircraft = aircraft
        self.thrust_scale = self.aircraft.aircraft_params['mass'] * 9.81 / \
            Config.HOVER_THROTTLE
            

    def compute_aoa(self, u:float, w:float) -> float:
        """
        Computes the angle of attack
        """
        #check divide by zero 
        # if u == 0:
        #     return 0.0
        
        #compute the angle of attack
        return np.arctan2(w, u)

    def compute_beta(self, u:float, v:float, w:float) -> float:
        """
        Computes the sideslip angle
        """
        #check divide by zero 
        # if u == 0:
        #     return 0.
        airspeed = np.sqrt(u**2 + v**2 + w**2)
        beta_rad = np.arcsin(v/u)
        return beta_rad
    
    def compute_moments(self,
                        input_aileron:float,
                        input_elevator:float,
                        input_rudder:float,
                        force:np.ndarray, 
                        states:np.ndarray) -> np.ndarray:
        """
        Returns the moments in the body frame of reference

        Parameters
        ----------
        input_aileron : float
            The aileron input in radians
        input_elevator : float
            The elevator input in radians   
        input_rudder : float
            The rudder input in radians
        force : np.ndarray
            The force vector in the body frame of reference

        Returns
        -------
        np.ndarray
            The moment vector in the body frame of reference
        """
        u = states[3]
        v = states[4]
        w = states[5]

        alpha = self.compute_aoa(u,w)
        beta = self.compute_beta(u,v,w)

        #take sqrt of u^2 + v^2 + w^2 and make sure its not nan
        airspeed = np.linalg.norm(states[3:6])
        if np.isnan(airspeed):
            airspeed = 0.0

        p = states[9]
        q = states[10]
        r = states[11]

        effective_airspeed = airspeed
        coefficient = self.aircraft.aircraft_params
        s = coefficient["s"]
        c = coefficient["c"]
        b = coefficient["b"]
        c_l_0 = coefficient["c_l_0"]
        c_l_b = coefficient["c_l_b"]
        c_l_p = coefficient["c_l_p"]
        c_l_r = coefficient["c_l_r"]
        c_l_deltaa = coefficient["c_l_deltaa"]
        c_l_deltar = coefficient["c_l_deltar"]
        c_m_0 = coefficient["c_m_0"]
        c_m_a = coefficient["c_m_a"]
        c_m_q = coefficient["c_m_q"]
        c_m_deltae = coefficient["c_m_deltae"]
        c_n_0 = coefficient["c_n_0"]
        c_n_b = coefficient["c_n_b"]
        c_n_p = coefficient["c_n_p"]
        c_n_r = coefficient["c_n_r"]
        c_n_deltaa = coefficient["c_n_deltaa"]
        c_n_deltar = coefficient["c_n_deltar"]
        CGOffset_x = coefficient["CGOffset_x"]
        CGOffset_y = coefficient["CGOffset_y"]
        CGOffset_z = coefficient["CGOffset_z"]

        CGOffset = [CGOffset_x, CGOffset_y, CGOffset_z]

        qbar = 0.5 * self.aircraft.rho * effective_airspeed**2 * s

        if effective_airspeed == 0:
            la, ma, na = 0, 0, 0
        else:
            la = qbar * b * (c_l_0 + c_l_b * beta +
                            c_l_p * b * p / (2 * effective_airspeed) +
                            c_l_r * b * r / (2 * effective_airspeed) +
                            c_l_deltaa * input_aileron +
                            c_l_deltar * input_rudder)

            ma = qbar * c * (c_m_0 + c_m_a * alpha +
                             c_m_q * c * q / (2 * effective_airspeed) +
                             c_m_deltae * input_elevator)

            na = qbar * b * (c_n_0 + c_n_b * beta + c_n_p * b * p / (2 * effective_airspeed) +
                        c_n_r * b * r / (2 * effective_airspeed) +
                        c_n_deltaa * input_aileron +
                        c_n_deltar * input_rudder)
            
        la +=  CGOffset[1] * force[2] - CGOffset[2] * force[1]
        ma += -CGOffset[0] * force[2] + CGOffset[2] * force[0]
        na += -CGOffset[1] * force[0] + CGOffset[0] * force[1]

        moments = np.array([la, ma, na]) 
        return moments
    
    def drag_coeff(self, alpha_rad)-> float:
        """
        computes the induced drag coefficient with a linear and flat plate model
        https://www.grc.nasa.gov/www/k-12/VirtualAero/BottleRocket/airplane/induced.html
        """
        coefficient = self.aircraft.aircraft_params
        b = coefficient["b"]
        s = coefficient["s"]
        c_drag_p = coefficient["c_drag_p"]
        c_lift_0 = coefficient["c_lift_0"]
        c_lift_a0 = coefficient["c_lift_a"]
        oswald = coefficient["oswald"]
        
        ar = pow(b, 2) / s
        c_drag_a = c_drag_p + pow(c_lift_0 + c_lift_a0 * alpha_rad, 2) / (np.pi * oswald * ar)

        return c_drag_a
    
    ### advancing translation quantities
    def lift_coeff(self, alpha:float) -> float:
        """
        Computes the lift coefficient with a linear and flat plate model
        """
        coefficient = self.aircraft.aircraft_params
        alpha0 = coefficient["alpha_stall"]
        M = coefficient["mcoeff"]
        c_lift_0 = coefficient["c_lift_0"]
        c_lift_a0 = coefficient["c_lift_a"]

        max_alpha_delta = 0.8
        if alpha - alpha0 > max_alpha_delta:
            alpha = alpha0 + max_alpha_delta
        elif alpha0 - alpha > max_alpha_delta:
            alpha = alpha0 - max_alpha_delta

        sigmoid = (1 + np.exp(-M * (alpha - alpha0)) + \
                   np.exp(M * (alpha + alpha0))) / (1 + math.exp(-M * (alpha - alpha0))) \
                    / (1 + math.exp(M * (alpha + alpha0)))
        linear = (1.0 - sigmoid) * (c_lift_0 + c_lift_a0 * alpha)  # Lift at small AoA
        flat_plate = sigmoid * (2 * math.copysign(1, alpha) * math.pow(math.sin(alpha), 2) * math.cos(alpha))  # Lift beyond stall

        return linear + flat_plate
    
    def compute_forces(self, input_aileron_rad:float,
                       input_elevator_rad:float,
                       input_rudder_rad:float,
                       input_thrust:float, 
                       states:np.ndarray) -> np.ndarray:
        """
        Computes the forces in the body frame of reference
        returns the forces in the body frame of reference
        """
        u = states[3]
        v = states[4]
        w = states[5]
        alpha_rad = self.compute_aoa(u,w)
        beta_rad = self.compute_beta(u,v,w)

        airspeed = np.linalg.norm(states[3:6])
        if np.isnan(airspeed):
            airspeed = 0.0
        coefficient = self.aircraft.aircraft_params
        
        qbar = 0.5*self.aircraft.rho*airspeed**2

        p = states[9]
        q = states[10]
        r = states[11]

        c_drag_q = coefficient["c_drag_q"]
        c_lift_q = coefficient["c_lift_q"]
        s = coefficient["s"]
        c = coefficient["c"]
        b = coefficient["b"]
        c_drag_deltae = coefficient["c_drag_deltae"]
        c_lift_deltae = coefficient["c_lift_deltae"]
        c_y_0 = coefficient["c_y_0"]
        c_y_b = coefficient["c_y_b"]
        c_y_p = coefficient["c_y_p"]
        c_y_r = coefficient["c_y_r"]
        c_y_deltaa = coefficient["c_y_deltaa"]
        c_y_deltar = coefficient["c_y_deltar"]

        # get lift and drag alpha coefficients 
        c_lift_a = self.lift_coeff(alpha_rad)
        c_drag_a = self.drag_coeff(alpha_rad)

        # coefficients to the body frame
        c_x_a = -c_drag_a*np.cos(alpha_rad) + c_lift_a*np.sin(alpha_rad)
        c_x_q = -c_drag_q*np.cos(alpha_rad) + c_lift_q*np.sin(alpha_rad)
        c_z_a = -c_drag_a*np.sin(alpha_rad) - c_lift_a*np.cos(alpha_rad)
        c_z_q = -c_drag_q*np.sin(alpha_rad) - c_lift_q*np.cos(alpha_rad)

        #check if airsped is close to zero
        if airspeed == 0:
            f_ax_b = 0 
            f_ay_b = 0
            f_az_b = 0
        else:
            f_ax_b = qbar*(c_x_a + c_x_q*c*q/(2*airspeed) - \
                       c_drag_deltae*np.cos(alpha_rad)*abs(input_elevator_rad) + \
                        c_lift_deltae*np.sin(alpha_rad)*input_elevator_rad)
            f_ay_b = qbar*(c_y_0 + c_y_b*beta_rad + c_y_p*b*p/(2*airspeed) + \
                       c_y_r*b*r/(2*airspeed) + c_y_deltaa*input_aileron_rad + c_y_deltar*input_rudder_rad)
            f_az_b = qbar*(c_z_a + c_z_q*c*q/(2*airspeed) - \
                 c_drag_deltae*np.sin(alpha_rad)*abs(input_elevator_rad) - \
                    c_lift_deltae*np.cos(alpha_rad)*input_elevator_rad)
        
        # scale the thrust to newtons
        input_thrust = input_thrust * self.thrust_scale
        
        f_total_body = np.array([f_ax_b+input_thrust, 
                                 f_ay_b, 
                                 f_az_b])

        # print("ftotal: ", f_total_body)
        return f_total_body

    def compute_ang_acc(self, moments:np.ndarray,
                        states:np.ndarray) -> np.ndarray:
        """
        Computes the angular acceleration of the aircraft
        """
        aircraft_params = self.aircraft.aircraft_params
        Ixx = aircraft_params["Ixx"]
        Iyy = aircraft_params["Iyy"]
        Izz = aircraft_params["Izz"]

        p = states[9]
        q = states[10]
        r = states[11]

        first_part = (Ixx - Iyy) * q * r 
        second_part = (Izz - Iyy) * p * q
        third_part = (Ixx - Izz) * p * q

        p_dot = (moments[0] + first_part) / Ixx
        q_dot = (moments[1] + second_part) / Iyy
        r_dot = (moments[2] + third_part) / Izz

        return np.array([p_dot, q_dot, r_dot])


    def compute_derivatives(self,
                         input_aileron_rad:float,
                         input_elevator_rad:float,
                         input_rudder_rad:float,
                         input_thrust_n:float,
                         states:np.ndarray) -> np.ndarray:
        """
        Computes the derivatives of the aircraft 
        """
        forces = self.compute_forces(input_aileron_rad,
                                     input_elevator_rad,
                                     input_rudder_rad,
                                     input_thrust_n, 
                                     states)

        moments = self.compute_moments(input_aileron_rad,
                                    input_elevator_rad,
                                    input_rudder_rad,
                                    forces,
                                    states)
    
        # compute angular accelerations 
        p_q_r_dot = self.compute_ang_acc(moments, states)    
        p_q_r_dot = np.clip(-Config.MAX_RADIAN, 
                            Config.MAX_RADIAN, 
                            p_q_r_dot)
        
        phi = states[6]
        theta = states[7]
        psi = states[8]

        B = compute_B_matrix(phi, theta, psi)
        
        # compute angular velocities
        #phi_theta_psi_dot = np.dot(B, current_ang_velocities)
        phi_theta_psi_dot = np.matmul(B, p_q_r_dot)

        g = 9.81

        current_attitudes = states[6:9]

        p = p_q_r_dot[0]
        q = p_q_r_dot[1]
        r = p_q_r_dot[2]
        u = states[3]
        v = states[4]
        w = states[5]

        gravity_body_frame = np.array([
            g*np.sin(current_attitudes[1]), 
            g*np.sin(current_attitudes[0])*np.cos(current_attitudes[1]), 
            g*np.cos(current_attitudes[0])*np.cos(current_attitudes[1])])

        mass = self.aircraft.aircraft_params['mass']
        
        #accelerations
        u_dot = forces[0]/mass + gravity_body_frame[0] - (q*w)  + (r*v)
        v_dot = forces[1]/mass + gravity_body_frame[1] - (r*u)  + (p*w)
        w_dot = forces[2]/mass + gravity_body_frame[2] - (p*v)  + (q*u)

        # u_dot = np.clip(-Config.ACCEL_LIM, Config.ACCEL_LIM, u_dot)
        # v_dot = np.clip(-Config.ACCEL_LIM, Config.ACCEL_LIM, v_dot)
        # w_dot = np.clip(-Config.ACCEL_LIM, Config.ACCEL_LIM, w_dot)
        
        #velocities
        dcm_body_to_inertial = euler_dcm_body_to_inertial(phi, theta, psi)
        body_vel = np.array([u, v, w])
        inertial_vel = np.matmul(dcm_body_to_inertial, body_vel) 
        x_dot = inertial_vel[0]
        y_dot = inertial_vel[1]
        z_dot = inertial_vel[2]


        states_dot = np.array([x_dot, y_dot, z_dot,
                            u_dot, v_dot, w_dot,
                            phi_theta_psi_dot[0], phi_theta_psi_dot[1], phi_theta_psi_dot[2],
                            p_q_r_dot[0], p_q_r_dot[1], p_q_r_dot[2]])
        
        
        return states_dot
    
    def add_noise_to_states(self, states:np.ndarray) -> np.ndarray:
        """
        Adds noise to the start of the aircraft  
        """
        deg_noise = np.deg2rad(1.0)
        dist_noise = 1.0

        states[0] += random.uniform(-dist_noise, dist_noise)
        states[1] += random.uniform(-dist_noise, dist_noise)
        states[2] += random.uniform(-dist_noise, dist_noise)
        states[3] += random.uniform(-dist_noise, dist_noise)
        states[4] += random.uniform(-dist_noise, dist_noise)
        states[5] += random.uniform(-dist_noise, dist_noise)
        states[6] += random.uniform(-deg_noise, deg_noise)
        states[7] += random.uniform(-deg_noise, deg_noise)
        states[8] += random.uniform(-deg_noise, deg_noise)
        states[9] += random.uniform(-deg_noise, deg_noise)
        states[10] += random.uniform(-deg_noise, deg_noise)
        states[11] += random.uniform(-deg_noise, deg_noise)

        return states

    def rk45(self, 
             input_aileron_rad:float,
             input_elevator_rad:float,
             input_rudder_rad:float,
             input_thrust_n:float,
             states:np.ndarray,
             delta_time:float, 
             use_noise:bool=False) -> np.ndarray:
        """
        Simulates the aircraft using the Runge-Kutta 4th order method 
        """
        #get the current states
        current_states = states

        #compute the derivatives
        k1 = delta_time * self.compute_derivatives(input_aileron_rad,
                                        input_elevator_rad,
                                        input_rudder_rad,
                                        input_thrust_n,
                                        current_states)
        
        k2 = delta_time * self.compute_derivatives(input_aileron_rad,
                                        input_elevator_rad,
                                        input_rudder_rad,
                                        input_thrust_n,
                                        current_states + k1/2)
        
        k3 = delta_time * self.compute_derivatives(input_aileron_rad,
                                        input_elevator_rad,
                                        input_rudder_rad,
                                        input_thrust_n,
                                        current_states + k2/2)
        
        k4 = delta_time * self.compute_derivatives(input_aileron_rad,
                                        input_elevator_rad,
                                        input_rudder_rad,
                                        input_thrust_n,
                                        current_states + k3)
        
        new_states = current_states + (k1 + 2*k2 + 2*k3 + k4) / 6
     
        if use_noise == True:
            new_states = self.add_noise_to_states(new_states)

        return new_states
    
    def eulers(self, 
             input_aileron_rad:float,
             input_elevator_rad:float,
             input_rudder_rad:float,
             input_thrust_n:float,
             states:np.ndarray,
             delta_time:float,
             use_noise:bool=False) -> np.ndarray:
        """
        Simulates the aircraft using the Euler's method 
        """
        #get the current states
        current_states = states

        #compute the derivatives
        derivatives = self.compute_derivatives(input_aileron_rad,
                                        input_elevator_rad,
                                        input_rudder_rad,
                                        input_thrust_n,
                                        current_states)

        new_states = current_states + (derivatives * delta_time)

        if use_noise == True:
            new_states = self.add_noise_to_states(new_states)

        return new_states
    

class AircraftCasadi():
    def __init__(self,aircraft_params:dict) -> None:
        self.define_states()
        self.define_controls()
        self.aircraft_params = aircraft_params
        # self.aircraft_params['mass'] = 5.0
        self.thrust_scale = self.aircraft_params['mass'] * 9.81 / \
            Config.HOVER_THROTTLE
            
        self.compute_forces() 
        self.compute_moments()


    def define_states(self):
        self.x = ca.MX.sym('x') # North position
        self.y = ca.MX.sym('y') # East position
        self.z = ca.MX.sym('z')

        self.u = ca.MX.sym('u') # Velocity along body x-axis
        self.v = ca.MX.sym('v')
        self.w = ca.MX.sym('w')

        self.phi = ca.MX.sym('phi') # Roll angle
        self.theta = ca.MX.sym('theta')
        self.psi = ca.MX.sym('psi')

        self.p = ca.MX.sym('p') # Roll rate
        self.q = ca.MX.sym('q')
        self.r = ca.MX.sym('r')

        self.states = ca.vertcat(
            self.x, self.y, self.z, 
            self.u, self.v, self.w, 
            self.phi, self.theta, self.psi, 
            self.p, self.q, self.r)
        
        self.num_states = self.states.size()[0]
        self.n_states = self.num_states

    def define_controls(self):
        self.delta_a = ca.MX.sym('delta_a') # Aileron
        self.delta_e = ca.MX.sym('delta_e') # Elevator
        self.delta_r = ca.MX.sym('delta_r') # Rudder
        self.delta_t = ca.MX.sym('delta_t') # Throttle

        self.controls = ca.vertcat(
            self.delta_a, 
            self.delta_e, 
            self.delta_r, 
            self.delta_t)
        
        self.num_controls = self.controls.size()[0]
        self.n_controls = self.num_controls

    def compute_aoa(self) -> ca.MX:
        # Compute the angle of attack
        # check divide by zero
        alpha = ca.if_else(self.u < 1e-2, 0.0 , ca.atan2(self.w, self.u))
        return alpha

    def compute_beta(self) -> ca.MX:
        # Compute the sideslip angle
        # check divide by zero
        airspeed = ca.sqrt(self.u**2 + self.v**2 + self.w**2)
        beta = ca.if_else(self.u < 1e-2, 0.0, ca.arcsin(self.v/airspeed))
        # return ca.atan2(self.v, self.u)
        return beta

    def drag_coeff(self, ca_alpha)-> ca.MX:
        # Compute the drag coefficient
        coefficient = self.aircraft_params
        b = coefficient["b"]
        s = coefficient["s"]
        c_drag_p = coefficient["c_drag_p"]
        c_lift_0 = coefficient["c_lift_0"]
        c_lift_a0 = coefficient["c_lift_a"]
        oswald = coefficient["oswald"]
        
        ar = pow(b, 2) / s
        c_drag_a = c_drag_p + pow(c_lift_0 + c_lift_a0 * ca_alpha, 2) / (ca.pi * oswald * ar)

        return c_drag_a
    

    def lift_coeff(self, ca_alpha)-> ca.MX:
        """
        Computes the lift coefficient of the aircraft
        """
        coefficient = self.aircraft_params
        alpha0 = coefficient["alpha_stall"]
        M = coefficient["mcoeff"]
        c_lift_0 = coefficient["c_lift_0"]
        c_lift_a0 = coefficient["c_lift_a"]
        alpha = ca_alpha

        alpha_diff = alpha - alpha0
        max_alpha_delta = 0.8
        alpha = ca.if_else(alpha_diff > max_alpha_delta, 
                                alpha0+max_alpha_delta, 
                                alpha)
        
        other_way = alpha0 - alpha
        alpha = ca.if_else(other_way > max_alpha_delta,
                                alpha0-max_alpha_delta,
                                alpha)
        
        #use casadi to compute the sigmoid function
        sigmoid = (1 + ca.exp(-M * (alpha - alpha0)) + \
                     ca.exp(M * (alpha + alpha0))) / (1 + ca.exp(-M * (alpha - alpha0))) \
                        / (1 + ca.exp(M * (alpha + alpha0)))
        

        linear = (1.0 - sigmoid) * (c_lift_0 + c_lift_a0 * alpha)  # Lift at small AoA

        #use casadi to compute the flat plate function
        flat_plate = sigmoid * (2 * ca.sign(alpha) * ca.sin(alpha)**2 * ca.cos(alpha))
    
        return linear + flat_plate
    
    def compute_moments(self) -> ca.MX:
        alpha = self.compute_aoa()
        beta = self.compute_beta()
        coefficient = self.aircraft_params
        s = coefficient["s"]
        c = coefficient["c"]
        b = coefficient["b"]
        c_l_0 = coefficient["c_l_0"]
        c_l_b = coefficient["c_l_b"]
        c_l_p = coefficient["c_l_p"]
        c_l_r = coefficient["c_l_r"]
        c_l_deltaa = coefficient["c_l_deltaa"]
        c_l_deltar = coefficient["c_l_deltar"]
        c_m_0 = coefficient["c_m_0"]
        c_m_a = coefficient["c_m_a"]
        c_m_q = coefficient["c_m_q"]
        c_m_deltae = coefficient["c_m_deltae"]
        c_n_0 = coefficient["c_n_0"]
        c_n_b = coefficient["c_n_b"]
        c_n_p = coefficient["c_n_p"]
        c_n_r = coefficient["c_n_r"]
        c_n_deltaa = coefficient["c_n_deltaa"]
        c_n_deltar = coefficient["c_n_deltar"]
        CGOffset_x = coefficient["CGOffset_x"]
        CGOffset_y = coefficient["CGOffset_y"]
        CGOffset_z = coefficient["CGOffset_z"]

        CGOffset = ca.vertcat(CGOffset_x, CGOffset_y, CGOffset_z)
        CGOffset = ca.reshape(CGOffset, (3, 1))

        rho = 1.225

        airspeed = ca.sqrt(self.u**2 + self.v**2 + self.w**2)
        effective_airspeed = airspeed
        qbar = 0.5 * rho * effective_airspeed**2

        input_aileron = self.delta_a
        input_rudder = self.delta_r
        input_elevator = self.delta_e

        la = qbar * b * (c_l_0 + c_l_b * beta +
                        c_l_p * b * self.p / (2 * effective_airspeed) +
                        c_l_r * b * self.r / (2 * effective_airspeed) +
                        c_l_deltaa * input_aileron +
                        c_l_deltar * input_rudder)

        ma = qbar * c * (c_m_0 + c_m_a * alpha +
                            c_m_q * c * self.q / (2 * effective_airspeed) +
                            c_m_deltae * input_elevator)

        na = qbar * b * (c_n_0 + c_n_b * beta + c_n_p * b * self.p / (2 * effective_airspeed) +
                    c_n_r * b * self.r / (2 * effective_airspeed) +
                    c_n_deltaa * input_aileron +
                    c_n_deltar * input_rudder)
        

        la = ca.if_else(airspeed == 0, 0, la)
        ma = ca.if_else(airspeed == 0, 0, ma)
        na = ca.if_else(airspeed == 0, 0, na)

        #add offset to moments
        force = self.force_function(self.states, self.controls)
        la += CGOffset[1] * force[2] - CGOffset[2] * force[1]
        ma += -CGOffset[0] * force[2] + CGOffset[2] * force[0]
        na += -CGOffset[1] * force[0] + CGOffset[0] * force[1]

        moments = ca.vertcat(la, ma, na)

        self.moment_function = ca.Function('compute_moments',
                                        [self.states, self.controls],
                                        [moments],
                                        ['states', 'controls'], ['moments'])
        
        return moments
    
    def compute_forces(self) -> ca.MX:
        alpha = self.compute_aoa()
        beta = self.compute_beta()
        airspeed = ca.sqrt(self.u**2 + self.v**2 + self.w**2)
        p = self.p
        q = self.q
        r = self.r
        delta_e = self.delta_e
        delta_a = self.delta_a
        delta_r = self.delta_r
        delta_t = self.delta_t

        qbar = 0.5 * 1.225 * airspeed**2
        coefficient = self.aircraft_params

        c_drag_q = coefficient["c_drag_q"]
        c_lift_q = coefficient["c_lift_q"]
        s = coefficient["s"]
        c = coefficient["c"]
        b = coefficient["b"]
        c_drag_deltae = coefficient["c_drag_deltae"]
        c_lift_deltae = coefficient["c_lift_deltae"]
        c_y_0 = coefficient["c_y_0"]
        c_y_b = coefficient["c_y_b"]
        c_y_p = coefficient["c_y_p"]
        c_y_r = coefficient["c_y_r"]
        c_y_deltaa = coefficient["c_y_deltaa"]
        c_y_deltar = coefficient["c_y_deltar"]

        # get lift and drag alpha coefficients 
        c_lift_a = self.lift_coeff(alpha)
        c_drag_a = self.drag_coeff(alpha)

        # coefficients to the body frame
        c_x_a = -c_drag_a*ca.cos(alpha) + c_lift_a*ca.sin(alpha)
        c_x_q = -c_drag_q*ca.cos(alpha) + c_lift_q*ca.sin(alpha)
        c_z_a = -c_drag_a*ca.sin(alpha) - c_lift_a*ca.cos(alpha)
        c_z_q = -c_drag_q*ca.sin(alpha) - c_lift_q*ca.cos(alpha)

        f_ax_b = qbar * (c_x_a + c_x_q * c*q / (2 * airspeed) - \
                         c_drag_deltae * ca.cos(alpha) * delta_e +
                        c_lift_deltae * ca.sin(alpha) * delta_e)
        
        f_ay_b = qbar * (c_y_0 + c_y_b * beta + c_y_p * b * p / (2 * airspeed) + \
                         c_y_r * b * r / (2 * airspeed) +
                        c_y_deltaa * delta_a + c_y_deltar * delta_r)
        
        f_az_b = qbar*(c_z_a + c_z_q*c*q/(2*airspeed) - \
                c_drag_deltae*np.sin(alpha)*ca.fabs(delta_e) - \
                c_lift_deltae*np.cos(alpha)*delta_e)

        # Set forces to zero when airspeed is zero
        f_ax_b = ca.if_else(airspeed <= 1e-2, 0.0, f_ax_b)
        f_ay_b = ca.if_else(airspeed <= 1e-2, 0.0, f_ay_b)
        f_az_b = ca.if_else(airspeed <= 1e-2, 0.0, f_az_b)

        # Define the total force in the body frame
        
        #scale the thrust to newtons
        delta_t = delta_t * self.thrust_scale
        
        f_total_body = ca.vertcat(f_ax_b + delta_t, f_ay_b, f_az_b)                    
        

        self.force_function = ca.Function('compute_forces', 
                                        [self.states, self.controls], 
                                        [f_total_body],
                                        ['states', 'controls'], ['f_total_body'])   

        return f_total_body


    def compute_ang_acc(self, moments) -> ca.MX:
        """
        Computes the angular acceleration of the aircraft
        """
        Ixx = self.aircraft_params["Ixx"]
        Iyy = self.aircraft_params["Iyy"]
        Izz = self.aircraft_params["Izz"]

        first_part = (Ixx - Iyy) * self.q * self.r 
        second_part = (Izz - Iyy) * self.p * self.q
        third_part = (Ixx - Izz) * self.p * self.q

        p_dot = (moments[0] + first_part) / Ixx
        q_dot = (moments[1] + second_part) / Iyy
        r_dot = (moments[2] + third_part) / Izz

        return ca.vertcat(p_dot, q_dot, r_dot)
        
    def set_state_space(self) -> ca.Function:

        moments = self.moment_function(self.states, self.controls)
        sim_forces = self.force_function(self.states, self.controls)

        Ixx = self.aircraft_params["Ixx"]
        Iyy = self.aircraft_params["Iyy"]
        Izz = self.aircraft_params["Izz"]

        first_part = (Ixx - Iyy) * self.q * self.r 
        second_part = (Izz - Iyy) * self.p * self.q
        third_part = (Ixx - Izz) * self.p * self.q

        self.p_dot = (moments[0] + first_part) / Ixx
        self.q_dot = (moments[1] + second_part) / Iyy
        self.r_dot = (moments[2] + third_part) / Izz

        g = 9.81

        mass = self.aircraft_params["mass"]
        print("mass is: ", mass)
        B = ca_compute_B_matrix(self.phi, self.theta, self.psi)
        ang_vel_bf = ca.vertcat(self.p, self.q, self.r)
        attitudes = ca.mtimes(B, ang_vel_bf)

        self.phi_dot = attitudes[0]
        self.theta_dot = attitudes[1]
        self.psi_dot = attitudes[2]

        gravity_body_frame = ca.vertcat(
            g*ca.sin(self.theta),
            g*ca.sin(self.phi)*ca.cos(self.theta),
            g*ca.cos(self.phi)*ca.cos(self.theta)
        )
        
        self.u_dot = sim_forces[0]/mass - gravity_body_frame[0] - \
            (self.q*self.w) + (self.r*self.v)
        
        self.v_dot = sim_forces[1]/mass + gravity_body_frame[1] - \
            (self.r*self.u) + (self.p*self.w)
        
        self.w_dot = sim_forces[2]/mass + gravity_body_frame[2] - \
            (self.p*self.v) + (self.q*self.u)

        # #get x, y, z in inertial frame
        dcm_body_to_inertial = ca_euler_dcm_body_to_inertial(
            self.phi, self.theta, self.psi)
        body_vel = ca.vertcat(self.u, self.v, self.w)
        inertial_vel = ca.mtimes(dcm_body_to_inertial, body_vel)
         
        self.x_dot = inertial_vel[0]
        self.y_dot = inertial_vel[1]
        self.z_dot = inertial_vel[2]

        self.states_dot = ca.vertcat(
            self.x_dot, self.y_dot, self.z_dot, 
            self.u_dot, self.v_dot, self.w_dot, 
            self.phi_dot, self.theta_dot, self.psi_dot, 
            self.p_dot, self.q_dot, self.r_dot)
        

        #right hand side of the state space equation
        self.f = ca.Function('f', [self.states, self.controls], 
                                [self.states_dot])
    
        return self.f 
    
class LonAirPlane():
    """
    Linearized equations from this resource
    https://control.asu.edu/Classes/MMAE441 /Aircraft/441Lecture10.pdf
    """
    def __init__(self, aircraft_params:dict) -> None:
        self.aircraft_params = aircraft_params
    
    def lift_coeff(self, alpha:float) -> float:
        """
        Computes the lift coefficient with a linear and flat plate model
        """
        coefficient = self.aircraft_params
        alpha0 = coefficient["alpha_stall"]
        M = coefficient["mcoeff"]
        c_lift_0 = coefficient["c_lift_0"]
        c_lift_a0 = coefficient["c_lift_a"]

        max_alpha_delta = 0.8
        if alpha - alpha0 > max_alpha_delta:
            alpha = alpha0 + max_alpha_delta
        elif alpha0 - alpha > max_alpha_delta:
            alpha = alpha0 - max_alpha_delta

        sigmoid = (1 + np.exp(-M * (alpha - alpha0)) + \
                   np.exp(M * (alpha + alpha0))) / (1 + math.exp(-M * (alpha - alpha0))) \
                    / (1 + math.exp(M * (alpha + alpha0)))
        linear = (1.0 - sigmoid) * (c_lift_0 + c_lift_a0 * alpha)  # Lift at small AoA
        flat_plate = sigmoid * (2 * math.copysign(1, alpha) * math.pow(math.sin(alpha), 2) * math.cos(alpha))  # Lift beyond stall

        return linear + flat_plate
    
    def drag_coeff(self, alpha:float) -> float:
        """
        computes the induced drag coefficient with a linear and flat plate model
        https://www.grc.nasa.gov/www/k-12/VirtualAero/BottleRocket/airplane/induced.html
        """
        coefficient = self.aircraft_params
        b = coefficient["b"]
        s = coefficient["s"]
        c_drag_p = coefficient["c_drag_p"]
        c_lift_0 = coefficient["c_lift_0"]
        c_lift_a0 = coefficient["c_lift_a"]
        oswald = coefficient["oswald"]
        
        ar = pow(b, 2) / s
        c_drag_a = c_drag_p + pow(c_lift_0 + c_lift_a0 * alpha, 2) / (np.pi * oswald * ar)

        return c_drag_a
    
    def compute_A(self, u_airspeed_ms:float,
                  theta_rad:float, 
                  use_w:bool=False,
                  w:float=0.0) -> None:
        """
        Computes the A matrix for the aircraft
        without the speed term should be a 5 x 5 matrix
        
        States are as follows:
        [u, w, q, theta, z]
        
        Takes in the body u velocity in m/s and the pitch angle in radians
        Pitch angle is in radians with up being positive in the body frame
        """
        
        RHO = Config.RHO
        G = Config.G
        Q = 0.5 * RHO * u_airspeed_ms**2
        m = self.aircraft_params['mass']
        s = self.aircraft_params['s']
        c = self.aircraft_params['c']
        b = self.aircraft_params['b']
        
        # Ixx = self.aircraft_params['Ixx']
        Iyy = self.aircraft_params['Iyy']
        # Izz = self.aircraft_params['Izz']
        
        # c_m_0 = self.aircraft_params['c_m_0']
        c_m_a = self.aircraft_params['c_m_a']
        c_m_q = self.aircraft_params['c_m_q']
        # c_m_deltae = self.aircraft_params['c_m_deltae']
        
        c_lift_0 = self.aircraft_params['c_lift_0']
        
        if 'c_drag_0' in self.aircraft_params:
            c_drag_0 = self.aircraft_params['c_drag_0']
        else:
            c_drag_0 = 0.0
        # c_drag_a = self.aircraft_params['c_drag_a']
        
        if 'c_lift_u' not in self.aircraft_params:
            c_lift_u = 0.0
        else:
            c_lift_u = self.aircraft_params['c_lift_u']
        
        c_lift_a = self.aircraft_params['c_lift_a']
        
        #aspect ratio
        ar = np.power(b,2)/s
        
        c_drag_p = self.aircraft_params['c_drag_p']
    
        #check if oswald exists in the dictionary
        if 'oswald' in self.aircraft_params:
            oswald = self.aircraft_params['oswald']
        else:
            oswald = 0.7
    
        s_theta = np.sin(theta_rad)
        c_theta = np.cos(theta_rad)
    
        if use_w == True:
            constant = (u_airspeed_ms*c_theta) #+ (w*c_theta)
            constant2 = (u_airspeed_ms*s_theta) #+ (w*s_theta)
            alpha = np.arctan2(w, u_airspeed_ms)
        else:
            constant = u_airspeed_ms*c_theta
            constant2 = u_airspeed_ms*s_theta
            alpha = theta_rad

        #lift coefficient 
        #C_l = (m*G / (Q*s)) * alpha
        C_l = self.lift_coeff(alpha)

        # c_drag_a = c_drag_p + np.power(c_lift_0 + c_lift_a*alpha,2) / \
        #     (np.pi * oswald * ar)
        k = 1/(np.pi) * oswald * ar
        # k = 1/np.pi * 6 * 0.7 # wtf is this 
        
        c_drag_a = k * (C_l)**2*alpha**2 + c_drag_0             
        #c_drag_a = self.drag_coeff(alpha)
        
        #check if c_drag_u exists in the dictionary
        if 'c_drag_u' in self.aircraft_params:
            c_drag_u = self.aircraft_params['c_drag_u']
        else:
            c_drag_u = 0.0
        
        #remind Austin to add the mass term in first spreadsheet
        X_u = -(c_drag_u + 2*c_drag_0) * Q * s / (m*u_airspeed_ms)
        X_w = -(c_drag_a - c_lift_0) * Q * s  / (m*u_airspeed_ms)
        
        Z_u = -(c_lift_u + (2*c_lift_0)) * Q * s / (m*u_airspeed_ms)
        Z_w = -(c_lift_a + c_drag_0) * Q * s / (m*u_airspeed_ms)
        Z_q = u_airspeed_ms #THIS IS WEIRD
                
        M_u = 0.0 
        M_w = (c_m_a * Q * s * c) / (Iyy*u_airspeed_ms)
        #M_q = (c_m_q) * c * Q * s * c / (2*Iyy*u_airspeed_ms*u_airspeed_ms)
        M_q = (c_m_q * c / (2 * u_airspeed_ms)) * (Q * s * c / Iyy);
        
        A = np.array([
            [X_u, X_w, 0,            -G*c_theta, 0 , 0], #u
            [Z_u, Z_w, Z_q,          -G*s_theta, 0,  0], #w
            [M_u, M_w, M_q,           0, 0, 0],          #q
            [0 ,  0,   1,             0, 0, 0],          #theta
            [-s_theta ,  c_theta,   0, -constant, 0, 0],  #z
            [c_theta,    s_theta,   0, -constant2, 0, 0]  #x
            ])
        
        return A 
    
    def get_X_u(self,A:np.ndarray) -> float:
        return A[0,0]
    
    def get_X_w(self,A:np.ndarray) -> float:
        return A[0,1]
    
    def get_X_q(self,A:np.ndarray) -> float:
        return A[0,2]   
    
    def get_eigenvalues(self, A:np.ndarray) -> np.ndarray:
        """
        Computes the eigenvalues of the matrix A
        """
        return np.linalg.eigvals(A)

    def compute_B(self, u_airspeed_ms:float) -> None:
        """
        Computes the static B matrix for the aircraft
        without the speed term should be a 5 x 2 matrix
        
        Takes in the body u velocity in m/s and the pitch angle in radians
        Pitch angle is in radians with up being positive in the body frame
        """
        s = self.aircraft_params['s']
        c = self.aircraft_params['c']

        Iyy = self.aircraft_params['Iyy']
        c_m_deltae = self.aircraft_params['c_m_deltae']
        c_lift_deltae = self.aircraft_params['c_lift_deltae']
        
        #remind Austin to add the mass term in first spreadsheet
        m = self.aircraft_params['mass']
        RHO = Config.RHO
        
        G = Config.G
        Q = 0.5 * RHO * u_airspeed_ms**2

        Z_deltathrust = 0.0                
        X_de = 0.0
        X_dt = 1/(m)
        
        Z_de = -Q * c_lift_deltae * s / m
        M_de = -c_m_deltae * (Q * s * c) /(Iyy) 
        
        B = np.array([
            [X_de, X_dt],
            [Z_de, Z_deltathrust],
            [M_de, 0],
            [0  ,  0],
            [0  ,  0],
            [0  ,  0]
            ]) 
        
        return B
        
    def compute_derivatives(self, 
                            input_elevator_rad:float,
                            input_thrust_n:float, 
                            states:np.ndarray,
                            A:np.ndarray,
                            B:np.ndarray) -> np.ndarray:
        
        thrust_scale = self.aircraft_params['mass'] * 9.81 / \
            Config.HOVER_THROTTLE
            
        input_thrust_n = input_thrust_n * thrust_scale
        
        controls = np.array([input_elevator_rad, input_thrust_n])
        
        x_dot = np.matmul(A, states) + np.matmul(B, controls)
        
        return x_dot
    
    def rk45(self,
             input_elevator_rad:float,
             input_thrust_n:float, 
             states:np.ndarray, A:np.ndarray,
             B:np.ndarray,
             delta_time:float) -> np.ndarray:
        """
        Simulates the aircraft using the Runge-Kutta 4th order method 
        """
        #get the current states
        current_states = states

        #compute the derivatives
        k1 = delta_time * self.compute_derivatives(input_elevator_rad,
                                        input_thrust_n,
                                        current_states, A, B)
        
        k2 = delta_time * self.compute_derivatives(input_elevator_rad,
                                        input_thrust_n,
                                        current_states + k1/2, A, B)
        
        k3 = delta_time * self.compute_derivatives(input_elevator_rad,
                                        input_thrust_n,
                                        current_states + k2/2, A, B)
        
        k4 = delta_time * self.compute_derivatives(input_elevator_rad,
                                        input_thrust_n,
                                        current_states + k3, A, B)
        
        new_states = current_states + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        return new_states
    
    
class LonAirPlaneCasadi():
    """
    Longitudinal aircraft model for use with casadi
    
    A = [
        X_u, X_w, 0, -g*cos(theta_0), 0;
        Z_u, Z_w, u_0, -g*sin(theta_0), 0;
        M_u, M_w, M_q, 0, 0;
        0, 0, 1, 0, 0;
        -sin(theta_0), cos(theta_0), 0, 0, 0
    ]
    
    B = [
        X_deltae X_deltathrust;
        Z_deltae 0;
        M_deltae 0;
        0 0;
        0 0
    ]
    
    u = [
        delta_e;
        delta_thrust
    ]
    
    Need to have a static A computed
    Have A be updated as A_dot
    
    Have B computed statically as well
    Have B be updated as B_dot
    
    Multiply Adot and Bdot with controls to get the
    derivatives of the states    
    """
    def __init__(self, aircraft_params:dict, 
                 use_own_A:bool=False,
                 A:np.ndarray=None,
                 use_own_B:bool=False,
                 B:np.ndarray=None) -> None:
        self.aircraft_params = aircraft_params
        #velocity trim condition for the aircraft
        self.define_states()
        self.define_controls()
        
        self.use_own_A = use_own_A
        self.use_own_B = use_own_B
        
        if use_own_A == False:
            self.compute_A()
        else:
            self.A = A
        
        if use_own_B == False:
            self.compute_B()
        else:
            self.B = B
        
    def define_states(self) -> None:
        self.u = ca.MX.sym('u')
        self.w = ca.MX.sym('w')
        self.q = ca.MX.sym('q')
        self.theta = ca.MX.sym('theta')
        self.h = ca.MX.sym('h')
        self.x = ca.MX.sym('x')
        
        self.states = ca.vertcat(self.u, 
                                 self.w, 
                                 self.q, 
                                 self.theta,
                                 self.h, 
                                 self.x)
        
        self.n_states = self.states.size()[0]
    
    def define_controls(self) -> None:
        
        self.thrust_scale = self.aircraft_params['mass'] * 9.81 / \
            Config.HOVER_THROTTLE
        self.delta_e = ca.MX.sym('delta_e')
        self.delta_thrust = ca.MX.sym('delta_thrust')
        self.controls = ca.vertcat(self.delta_e, 
                                   self.delta_thrust)
        
        self.n_controls = self.controls.size()[0]

    
    def drag_coeff(self, alpha_rad)-> float:
        """
        computes the induced drag coefficient with a linear and flat plate model
        https://www.grc.nasa.gov/www/k-12/VirtualAero/BottleRocket/airplane/induced.html
        """
        coefficient = self.aircraft_params
        b = coefficient["b"]
        s = coefficient["s"]
        c_drag_p = coefficient["c_drag_p"]
        c_lift_0 = coefficient["c_lift_0"]
        c_lift_a0 = coefficient["c_lift_a"]
        oswald = coefficient["oswald"]
        
        ar = pow(b, 2) / s
        c_drag_a = c_drag_p + pow(c_lift_0 + c_lift_a0 * alpha_rad, 2) / (np.pi * oswald * ar)

        return c_drag_a
    
    ### advancing translation quantities
    def lift_coeff(self, alpha:float) -> float:
        """
        Computes the lift coefficient with a linear and flat plate model
        """
        coefficient = self.aircraft_params
        alpha0 = coefficient["alpha_stall"]
        M = coefficient["mcoeff"]
        c_lift_0 = coefficient["c_lift_0"]
        c_lift_a0 = coefficient["c_lift_a"]

        max_alpha_delta = 0.8
        if alpha - alpha0 > max_alpha_delta:
            alpha = alpha0 + max_alpha_delta
        elif alpha0 - alpha > max_alpha_delta:
            alpha = alpha0 - max_alpha_delta

        sigmoid = (1 + np.exp(-M * (alpha - alpha0)) + \
                   np.exp(M * (alpha + alpha0))) / (1 + math.exp(-M * (alpha - alpha0))) \
                    / (1 + math.exp(M * (alpha + alpha0)))
        linear = (1.0 - sigmoid) * (c_lift_0 + c_lift_a0 * alpha)  # Lift at small AoA
        flat_plate = sigmoid * (2 * math.copysign(1, alpha) * math.pow(math.sin(alpha), 2) * math.cos(alpha))  # Lift beyond stall

        return linear + flat_plate
    
    def compute_A(self) -> None:
        
        u = self.states[0]
        w = self.states[1]
        theta = self.states[3]
    
        # alpha_rad = ca.if_else(u < 1e-2, 0.0 , ca.atan2(w, u))
        alpha_rad = ca.atan2(w, u)
        RHO = Config.RHO
        G = Config.G
        Q = 0.5 * RHO * u**2
        m = self.aircraft_params['mass']
        s = self.aircraft_params['s']
        c = self.aircraft_params['c']
        b = self.aircraft_params['b']
        
        # Ixx = self.aircraft_params['Ixx']
        Iyy = self.aircraft_params['Iyy']
        # Izz = self.aircraft_params['Izz']
        
        # c_m_0 = self.aircraft_params['c_m_0']
        c_m_a = self.aircraft_params['c_m_a']
        c_m_q = self.aircraft_params['c_m_q']
        # c_m_deltae = self.aircraft_params['c_m_deltae']
        c_lift_0 = self.aircraft_params['c_lift_0']
        
        if 'c_drag_0' in self.aircraft_params:
            c_drag_0 = self.aircraft_params['c_drag_0']
        else:
            c_drag_0 = 0.0
        # c_drag_a = self.aircraft_params['c_drag_a']
        
        if 'c_lift_u' not in self.aircraft_params:
            c_lift_u = 0.0
        else:
            c_lift_u = self.aircraft_params['c_lift_u']
                
        #check if oswald exists in the dictionary
        if 'oswald' in self.aircraft_params:
            oswald = self.aircraft_params['oswald']
        else:
            oswald = 0.7
    
        #aspect ratio
        ar = np.power(b,2)/s
        
        c_drag_p = self.aircraft_params['c_drag_p']

        #lift coefficient         
        c_lift_a = self.aircraft_params['c_lift_a']
                    
        #lift coefficient 
        C_l = (m*G / (Q*s)) * alpha_rad

        # c_drag_a = c_drag_p + np.power(c_lift_0 + c_lift_a*alpha,2) / \
        #     (np.pi * oswald * ar)
        k = 1/(np.pi) * oswald * ar
        # k = 1/np.pi * 6 * 0.7 # wtf is this 
        
        #c_drag_a = k * (C_l)**2          
        
        c_drag_a = self.drag_coeff(alpha_rad)
        
        #check if c_drag_u exists in the dictionary
        if 'c_drag_u' in self.aircraft_params:
            c_drag_u = self.aircraft_params['c_drag_u']
        else:
            c_drag_u = 0.0
        
        #remind Austin to add the mass term in first spreadsheet
        X_u = -(c_drag_u + (2*c_drag_0)) * Q * s / (m*u)
        X_w = -(c_drag_a - c_lift_0) * Q * s  / (m*u)
        X_q = 0.0
        
        Z_u = -(c_lift_u + (2*c_lift_0)) * Q * s / (m*u)
        Z_w = -(c_lift_a + c_drag_0) * Q * s / (m*u)
        Z_q = u #THIS IS WEIRD
                
        M_u = 0.0
        M_w = (c_m_a) * Q * s * c / (Iyy*u)
        M_q = (c_m_q * c / (2 * u)) * (Q * s * c / Iyy);
                
        c_theta = ca.cos(theta)
        s_theta = ca.sin(theta)

        constant = (u*c_theta) + (w*c_theta)
        constant2 = (u*s_theta) + (w*s_theta)
        #create a casadi 5 x 5 matrix
        A = ca.vertcat(
            ca.horzcat(X_u, X_w,  0,         -G*c_theta,    0, 0),
            ca.horzcat(Z_u, Z_w,  Z_q,       -G*s_theta,    0, 0),
            ca.horzcat(M_u, M_w,  M_q,        0,            0, 0),
            ca.horzcat(0 ,  0,    1,          0,            0, 0),
            ca.horzcat(-s_theta ,  c_theta,   0, constant,  0, 0),
            ca.horzcat(c_theta,    s_theta,   0, constant2, 0, 0)
            )
                
        self.A_function = ca.Function('compute_A',
                                    [self.states],
                                    [A],
                                    ['lon_states'], ['A'])
        
        return A
        
    def compute_B(self) -> None:
        """
        compute B matrices
        """
        u = self.states[0]
        self.thrust = self.controls[1] * self.thrust_scale
        s = self.aircraft_params['s']
        c = self.aircraft_params['c']

        Iyy = self.aircraft_params['Iyy']
        c_m_deltae = self.aircraft_params['c_m_deltae']
        c_lift_deltae = self.aircraft_params['c_lift_deltae']

        Q = 0.5 * Config.RHO * u**2
        
        m = self.aircraft_params['mass']
        
        Z_deltathrust = 0.0
        X_de = 0.0
        X_dt = 1/m
        Z_de = -Q * c_lift_deltae * s / m

        M_de = -c_m_deltae * (Q * s * c) /(Iyy) 
        
        B = ca.vertcat(
            ca.horzcat(X_de, X_dt),
            ca.horzcat(Z_de, Z_deltathrust),
            ca.horzcat(M_de, 0),
            ca.horzcat(0  ,  0),
            ca.horzcat(0  ,  0),
            ca.horzcat(0  ,  0))

        controls = ca.vertcat(self.delta_e, self.thrust)
        Bu = ca.mtimes(B, controls)  

        self.get_B = ca.Function('compute_B',
                                    [self.states, self.controls],
                                    [B],
                                    ['lon_states', 'controls'], ['B'])
        

        self.B_function = ca.Function('compute_Bu',
                                    [self.states, self.controls],
                                    [Bu],
                                    ['B_states', 'controls'], ['Bu'])
        
        return Bu
    

    def set_state_space(self) -> None:
        #map thrust to newtons 
        
        if self.use_own_A == False:
            A = self.A_function(self.states)
        else:
            # self.A = A
            A = self.A
            
        if self.use_own_B == False:
            Bu = self.B_function(self.states, self.controls)
            # B = self.B
            # #normalize thrust to newtons
            # thrust_scale =  self.aircraft_params['mass'] * 9.81 / \
            # Config.HOVER_THROTTLE
            # thrust = self.controls[1] * thrust_scale
            # controls = ca.vertcat(self.delta_e, thrust)
            #Bu = ca.mtimes(B, controls)
        else:
            B = self.B
            #normalize thrust to newtons
            thrust_scale =  self.aircraft_params['mass'] * 9.81 / \
            Config.HOVER_THROTTLE
            thrust = self.controls[1] * thrust_scale
            controls = ca.vertcat(self.delta_e, thrust)
            Bu = ca.mtimes(B, controls)
        #Bu = self.B_function(self.states, self.controls)
                        
        self.z_dot = ca.mtimes(A, self.states) + Bu
        self.f = ca.Function('lon_dynamics', [self.states, self.controls], 
                                    [self.z_dot])
        
class LatAirPlane():
    """
    A = [
        Y_v, Y_p, Y_r, 0, -g*cos(theta_0);
        L_v, L_p, L_r, 0, 0;
        N_v, N_p, N_r, 0, 0;
        0, 1, tan(theta_0), 0, 0;
        0, 0, sec(theta_0), 0, 0
        1, 0, 0, 0, 0,      u*cos(theta_0), 0
    ]
    
    B = [
        Y_deltaa Y_deltar;
        L_deltaa L_deltar;
        N_deltaa N_deltar;
        0 0;
        0 0,
        0 0
    ]
    
    u = [
        delta_a;
        delta_r
    ]
    
    states are as follows:
    [v, p, r, phi, psi, y]
    
    """
    def __init__(self, aircraft_params:dict) -> None:
        self.aircraft_params = aircraft_params
        
    def lift_coeff(self, alpha:float) -> float:
        """
        Computes the lift coefficient with a linear and flat plate model
        """
        coefficient = self.aircraft_params
        alpha0 = coefficient["alpha_stall"]
        M = coefficient["mcoeff"]
        c_lift_0 = coefficient["c_lift_0"]
        c_lift_a0 = coefficient["c_lift_a"]

        max_alpha_delta = 0.8
        if alpha - alpha0 > max_alpha_delta:
            alpha = alpha0 + max_alpha_delta
        elif alpha0 - alpha > max_alpha_delta:
            alpha = alpha0 - max_alpha_delta

        sigmoid = (1 + np.exp(-M * (alpha - alpha0)) + \
                   np.exp(M * (alpha + alpha0))) / (1 + math.exp(-M * (alpha - alpha0))) \
                    / (1 + math.exp(M * (alpha + alpha0)))
        linear = (1.0 - sigmoid) * (c_lift_0 + c_lift_a0 * alpha)  # Lift at small AoA
        flat_plate = sigmoid * (2 * math.copysign(1, alpha) * math.pow(math.sin(alpha), 2) * math.cos(alpha))  # Lift beyond stall

        return linear + flat_plate
    
    def drag_coeff(self, alpha:float) -> float:
        """
        computes the induced drag coefficient with a linear and flat plate model
        https://www.grc.nasa.gov/www/k-12/VirtualAero/BottleRocket/airplane/induced.html
        """
        coefficient = self.aircraft_params
        b = coefficient["b"]
        s = coefficient["s"]
        c_drag_p = coefficient["c_drag_p"]
        c_lift_0 = coefficient["c_lift_0"]
        c_lift_a0 = coefficient["c_lift_a"]
        oswald = coefficient["oswald"]
        
        ar = pow(b, 2) / s
        c_drag_a = c_drag_p + pow(c_lift_0 + c_lift_a0 * alpha, 2) / (np.pi * oswald * ar)

        return c_drag_a
    
    def compute_A(self, velocity:float, 
                  theta_rad:float) -> np.ndarray:
        
        Q = 0.5 * Config.RHO * velocity**2
        m = self.aircraft_params['mass']
        s = self.aircraft_params['s']
        b = self.aircraft_params['b']
        
        c_y_b = self.aircraft_params['c_y_b']
        c_y_p = self.aircraft_params['c_y_p']
        
        c_y_r = self.aircraft_params['c_y_r']
        c_l_b = self.aircraft_params['c_l_b']
        
        c_l_p = self.aircraft_params['c_l_p']
        c_l_r = self.aircraft_params['c_l_r']
        c_n_b = self.aircraft_params['c_n_b']
        c_n_p = self.aircraft_params['c_n_p']
        c_n_r = self.aircraft_params['c_n_r']
        c_n_deltar = self.aircraft_params['c_n_deltar']
        
        Ix = self.aircraft_params['Ixx']
        Iz = self.aircraft_params['Izz']
        
        
        Y_beta = Q * s * c_y_b / (m*velocity) 
        Y_p = Q * s * b * c_y_p / (2 * m * velocity)
        Y_r = Q * s * b * c_y_r / (2 * m * velocity)
        
        L_beta = Q * s * b * c_l_b / (Ix*velocity)
        L_p = Q * s * b**2 * c_l_p / (2 * Ix * velocity)
        L_r =  Q * s * b**2 * c_l_r / (2 * Ix  * velocity)

        N_beta = Q * s * b * c_n_b / (Iz*velocity);
        N_p = Q * s * b**2 * c_n_p / (2 * Iz * velocity)
        N_r = Q * s * b**2 * c_n_r / (2 * Iz * velocity)
        N_dr = Q * s * b * c_n_deltar / Iz;
        
        G = Config.G
        c_theta = np.cos(theta_rad)
        s_theta = np.sin(theta_rad)
        
        # c_phi = np.cos(phi_rad)
        # s_phi = np.sin(phi_rad)
        
        tan_theta = np.tan(theta_rad)
        sec_theta = 1 / c_theta
        
        A = np.array([
            [Y_beta,           Y_p,      -(velocity - Y_r),  (G*c_theta),      0,    0], #v
            [L_beta,           L_p,            L_r,           0,               0,    0], #p 
            [N_beta,           N_p,            N_r,           0,               0,    0], #r
            [0,                 1,             tan_theta,     0,               0,    0], #phi
            [0,                 0,             sec_theta,     0,               0,    0], #psi
            [1,                 0,             0,             velocity*c_theta,0,    0]  #y
            ])
        
        return A
    
    def compute_B(self, velocity:float) -> np.ndarray:
        """
        computes the B lateral matrix of the aircraft
        """
        Q = 0.5 * Config.RHO * velocity**2
        s = self.aircraft_params['s']
        b = self.aircraft_params['b']
        m = self.aircraft_params['mass']
        
        Ix = self.aircraft_params['Ixx']
        Iz = self.aircraft_params['Izz']
        
        c_l_da = self.aircraft_params['c_l_deltaa']
        c_y_dr = self.aircraft_params['c_y_deltar']
        
        c_l_dr = self.aircraft_params['c_l_deltar']
        c_n_da = self.aircraft_params['c_n_deltaa']
        c_n_dr = self.aircraft_params['c_n_deltar']
        
        L_da = Q * s * b * c_l_da / Ix;
        L_dr = Q * s * b * c_l_dr / Ix;
        
        N_da = Q * s * b * c_n_da / Iz;
        N_dr = Q * s * b * c_n_dr / Iz;
        
        Ydr = Q * s * c_y_dr / m;
        
        B = np.array([[0.0 , Ydr/velocity],
             [L_da, L_dr],
             [N_da, N_dr],
             [0.0 , 0.0], 
             [0.0 ,  0.0],
             [0.0 ,  0.0]])
        
        return B
    
    def compute_derivatives(self, 
                            input_aileron_rad:float,
                            input_rudder_rad:float, 
                            states:np.ndarray,
                            A:np.ndarray,
                            B:np.ndarray) -> np.ndarray:
        
        controls = np.array([input_aileron_rad, input_rudder_rad])
        
        x_dot = np.matmul(A, states) + np.matmul(B, controls)
        
        return x_dot
    
    def rk45(self,
                input_aileron_rad:float,
                input_rudder_rad:float, 
                states:np.ndarray, A:np.ndarray,
                B:np.ndarray,
                delta_time:float) -> np.ndarray:
            """
            Simulates the aircraft using the Runge-Kutta 4th order method 
            """
            #get the current states
            current_states = states
    
            #compute the derivatives
            k1 = delta_time * self.compute_derivatives(input_aileron_rad,
                                            input_rudder_rad,
                                            current_states, A, B)
            
            k2 = delta_time * self.compute_derivatives(input_aileron_rad,
                                            input_rudder_rad,
                                            current_states + k1/2, A, B)
            
            k3 = delta_time * self.compute_derivatives(input_aileron_rad,
                                            input_rudder_rad,
                                            current_states + k2/2, A, B)
            
            k4 = delta_time * self.compute_derivatives(input_aileron_rad,
                                            input_rudder_rad,
                                            current_states + k3, A, B)
            
            new_states = current_states + (k1 + 2*k2 + 2*k3 + k4) / 6
            
            return new_states
        
class LatAirPlaneCasadi():
    """
    Lateral aircraft model for use with casadi
    
    A = [
        Y_v, Y_p, Y_r, 0, -g*cos(theta_0);
        L_v, L_p, L_r, 0, 0;
        N_v, N_p, N_r, 0, 0;
        0, 1, tan(theta_0), 0, 0;
        0, 0, sec(theta_0), 0, 0
    ]
    
    B = [
        Y_deltaa Y_deltar;
        L_deltaa L_deltar;
        N_deltaa N_deltar;
        0 0;
        0 0
    ]
    
    u = [
        delta_a;
        delta_r
    ]
    
    Need to have a static A computed
    Have A be updated as A_dot
    
    Have B computed statically as well
    Have B be updated as B_dot
    
    Multiply Adot and Bdot with controls to get the
    derivatives of the states    
    """
    def __init__(self, aircraft_params:dict, 
                 use_own_A:bool=False,
                 A:np.ndarray=None,
                 use_own_B:bool=False,
                 B:np.ndarray=None) -> None:
        
        
        self.aircraft_params = aircraft_params
        #velocity trim condition for the aircraft
        self.define_states()
        self.define_controls()
        
        self.use_own_A = use_own_A
        self.use_own_B = use_own_B
        
        if use_own_A == False:
            self.compute_A()
        else:
            self.A = A
        
        if use_own_B == False:
            self.compute_B()
        else:
            self.B = B
        
    def define_states(self) -> None:
        self.v = ca.MX.sym('v')
        self.p = ca.MX.sym('p')
        self.r = ca.MX.sym('r')
        self.phi = ca.MX.sym('phi')
        self.psi = ca.MX.sym('psi')
        self.y = ca.MX.sym('y')
        
        self.states = ca.vertcat(self.v, 
                                 self.p, 
                                 self.r, 
                                 self.phi,
                                 self.psi,
                                 self.y)
        
        self.n_states = self.states.size()[0]
    
    def define_controls(self) -> None:
        self.delta_a = ca.MX.sym('delta_a')
        self.delta = ca.MX.sym('delta_r')
        self.controls = ca.vertcat(self.delta_a, 
                                   self.delta)
        
        self.n_controls = self.controls.size()[0]
        
    def set_state_space(self) -> None:
        A = self.A 
        B = self.B
        Bu = ca.mtimes(B, self.controls)
        
        self.z_dot = ca.mtimes(A, self.states) + Bu
        
        self.f = ca.Function('lat_dynamics', [self.states, self.controls],
                                    [self.z_dot])
        
        
class LinearizedAircraft():
    def __init__(self, aircraft_params:dict) -> None:
        self.aircraft_params = aircraft_params
        
    
    def lift_coeff(self, alpha:float) -> float:
        """
        Computes the lift coefficient with a linear and flat plate model
        """
        coefficient = self.aircraft_params
        alpha0 = coefficient["alpha_stall"]
        M = coefficient["mcoeff"]
        c_lift_0 = coefficient["c_lift_0"]
        c_lift_a0 = coefficient["c_lift_a"]

        max_alpha_delta = 0.8
        if alpha - alpha0 > max_alpha_delta:
            alpha = alpha0 + max_alpha_delta
        elif alpha0 - alpha > max_alpha_delta:
            alpha = alpha0 - max_alpha_delta

        sigmoid = (1 + np.exp(-M * (alpha - alpha0)) + \
                   np.exp(M * (alpha + alpha0))) / (1 + math.exp(-M * (alpha - alpha0))) \
                    / (1 + math.exp(M * (alpha + alpha0)))
        linear = (1.0 - sigmoid) * (c_lift_0 + c_lift_a0 * alpha)  # Lift at small AoA
        flat_plate = sigmoid * (2 * math.copysign(1, alpha) * math.pow(math.sin(alpha), 2) * math.cos(alpha))  # Lift beyond stall

        return linear + flat_plate
    
    def drag_coeff(self, alpha:float) -> float:
        """
        computes the induced drag coefficient with a linear and flat plate model
        https://www.grc.nasa.gov/www/k-12/VirtualAero/BottleRocket/airplane/induced.html
        """
        coefficient = self.aircraft_params
        b = coefficient["b"]
        s = coefficient["s"]
        c_drag_p = coefficient["c_drag_p"]
        c_lift_0 = coefficient["c_lift_0"]
        c_lift_a0 = coefficient["c_lift_a"]
        oswald = coefficient["oswald"]
        
        ar = pow(b, 2) / s
        c_drag_a = c_drag_p + pow(c_lift_0 + c_lift_a0 * alpha, 2) / (np.pi * oswald * ar)

        return c_drag_a
    
    def compute_A_lon(self, u_airspeed_ms:float,
                  theta_rad:float, 
                  use_w:bool=False,
                  w:float=0.0) -> np.ndarray:
        """
        Computes the A matrix for the aircraft
        without the speed term should be a 5 x 5 matrix
        
        States are as follows:
        [u, w, q, theta, z]
        
        Takes in the body u velocity in m/s and the pitch angle in radians
        Pitch angle is in radians with up being positive in the body frame
        """
        
        RHO = Config.RHO
        G = Config.G
        Q = 0.5 * RHO * u_airspeed_ms**2
        m = self.aircraft_params['mass']
        s = self.aircraft_params['s']
        c = self.aircraft_params['c']
        b = self.aircraft_params['b']
        
        # Ixx = self.aircraft_params['Ixx']
        Iyy = self.aircraft_params['Iyy']
        # Izz = self.aircraft_params['Izz']
        
        # c_m_0 = self.aircraft_params['c_m_0']
        c_m_a = self.aircraft_params['c_m_a']
        c_m_q = self.aircraft_params['c_m_q']
        # c_m_deltae = self.aircraft_params['c_m_deltae']
        
        c_lift_0 = self.aircraft_params['c_lift_0']
        
        if 'c_drag_0' in self.aircraft_params:
            c_drag_0 = self.aircraft_params['c_drag_0']
        else:
            c_drag_0 = 0.0
        # c_drag_a = self.aircraft_params['c_drag_a']
        
        if 'c_lift_u' not in self.aircraft_params:
            c_lift_u = 0.0
        else:
            c_lift_u = self.aircraft_params['c_lift_u']
        
        c_lift_a = self.aircraft_params['c_lift_a']
        
        #aspect ratio
        ar = np.power(b,2)/s
        
        c_drag_p = self.aircraft_params['c_drag_p']
    
        #check if oswald exists in the dictionary
        if 'oswald' in self.aircraft_params:
            oswald = self.aircraft_params['oswald']
        else:
            oswald = 0.7
    
        s_theta = np.sin(theta_rad)
        c_theta = np.cos(theta_rad)
    
        if use_w == True:
            constant = (u_airspeed_ms*c_theta) #+ (w*c_theta)
            constant2 = (u_airspeed_ms*s_theta) #+ (w*s_theta)
            alpha = np.arctan2(w, u_airspeed_ms)
        else:
            constant = u_airspeed_ms*c_theta
            constant2 = u_airspeed_ms*s_theta
            alpha = theta_rad

        #lift coefficient 
        #C_l = (m*G / (Q*s)) * alpha
        # C_l = self.lift_coeff(alpha)

        # c_drag_a = c_drag_p + np.power(c_lift_0 + c_lift_a*alpha,2) / \
        #     (np.pi * oswald * ar)
        # k = 1/(np.pi) * oswald * ar
        # k = 1/np.pi * 6 * 0.7 # wtf is this 
        
        # c_drag_a = k * (C_l)**2             
        c_drag_a = self.drag_coeff(alpha)
        
        #check if c_drag_u exists in the dictionary
        if 'c_drag_u' in self.aircraft_params:
            c_drag_u = self.aircraft_params['c_drag_u']
        else:
            c_drag_u = 0.0
        
        #remind Austin to add the mass term in first spreadsheet
        X_u = -(c_drag_u + 2*c_drag_0) * Q * s / (m*u_airspeed_ms)
        X_w = -(c_drag_a - c_lift_0) * Q * s  / (m*u_airspeed_ms)
        
        Z_u = -(c_lift_u + (2*c_lift_0)) * Q * s / (m*u_airspeed_ms)
        Z_w = -(c_lift_a + c_drag_0) * Q * s / (m*u_airspeed_ms)
        Z_q = u_airspeed_ms #THIS IS WEIRD
                
        M_u = 0.0 
        M_w = (c_m_a * Q * s * c) / (Iyy*u_airspeed_ms)
        #M_q = (c_m_q) * c * Q * s * c / (2*Iyy*u_airspeed_ms*u_airspeed_ms)
        M_q = (c_m_q * c / (2 * u_airspeed_ms)) * (Q * s * c / Iyy);
        
        c_theta = np.cos(theta_rad)
        s_theta = np.sin(theta_rad)
        
        A = np.array([
            [X_u, X_w, 0,          -G*c_theta, 0 , 0], #u
            [Z_u, Z_w, Z_q,        -G*s_theta, 0,  0], #w
            [M_u, M_w, M_q,         0, 0, 0],          #q
            [0 ,  0,   1,           0, 0, 0],          #theta
            [-s_theta ,  c_theta,   0, -constant, 0, 0],  #z
            [c_theta,    s_theta,   0, -constant2, 0, 0]  #x
            ])
        
        # extend 6 more columns for the lateral states  
        A_lon = np.hstack((A, np.zeros((A.shape[0], 6))))
        
        return A_lon
        
    def compute_A_lat(self, velocity:float,
                      theta_rad:float) -> np.ndarray:
        
        Q = 0.5 * Config.RHO * velocity**2
        m = self.aircraft_params['mass']
        s = self.aircraft_params['s']
        b = self.aircraft_params['b']
        
        c_y_b = self.aircraft_params['c_y_b']
        c_y_p = self.aircraft_params['c_y_p']
        
        c_y_r = self.aircraft_params['c_y_r']
        c_l_b = self.aircraft_params['c_l_b']
        
        c_l_p = self.aircraft_params['c_l_p']
        c_l_r = self.aircraft_params['c_l_r']
        c_n_b = self.aircraft_params['c_n_b']
        c_n_p = self.aircraft_params['c_n_p']
        c_n_r = self.aircraft_params['c_n_r']
        c_n_deltar = self.aircraft_params['c_n_deltar']
        
        Ix = self.aircraft_params['Ixx']
        Iz = self.aircraft_params['Izz']
        
        Y_beta = Q * s * c_y_b / (m*velocity)
        Y_p = Q * s * b * c_y_p / (2 * m * velocity)
        Y_r = Q * s * b * c_y_r / (2 * m * velocity)
        
        L_beta = Q * s * b * c_l_b / (Ix*velocity)
        L_p = Q * s * b**2 * c_l_p / (2 * Ix * velocity)
        L_r =  Q * s * b**2 * c_l_r / (2 * Ix  * velocity)

        N_beta = Q * s * b * c_n_b / (Iz*velocity);
        N_p = Q * s * b**2 * c_n_p / (2 * Iz * velocity)
        N_r = Q * s * b**2 * c_n_r / (2 * Iz * velocity)
        N_dr = Q * s * b * c_n_deltar / Iz;
        
        G = Config.G
        c_theta = np.cos(theta_rad)
        s_theta = np.sin(theta_rad)
        
        # c_phi = np.cos(phi_rad)
        # s_phi = np.sin(phi_rad)
        
        tan_theta = np.tan(theta_rad)
        sec_theta = 1 / c_theta
        
        A = np.array([
            [Y_beta,           Y_p,      -(velocity - Y_r),  (G*c_theta),      0,    0], #v
            [L_beta,           L_p,            L_r,           0,               0,    0], #p 
            [N_beta,           N_p,            N_r,           0,               0,    0], #r
            [0,                 1,             tan_theta,     0,               0,    0], #phi
            [0,                 0,             sec_theta,     0,               0,    0], #psi
            [1,                 0,             0,             0,               velocity*c_theta,    0]  #y
            ])

        A_lat = np.hstack((np.zeros((A.shape[0], 6)), A))        

        return A_lat        

    def compute_A_full(self, airspeed:float, theta_rad:float,
                   w_speed:float) -> np.ndarray:
        A_lon = self.compute_A_lon(airspeed, theta_rad, True, w_speed)
        A_lat = self.compute_A_lat(airspeed, theta_rad)
        
        #combine the two matrices
        A_full = np.vstack((A_lon, A_lat))
        
        return A_full
    
    def compute_B_lon(self, u_airspeed_ms:float) -> np.ndarray:
        """
        Computes the static B matrix for the aircraft
        without the speed term should be a 5 x 2 matrix
        
        Takes in the body u velocity in m/s and the pitch angle in radians
        Pitch angle is in radians with up being positive in the body frame
        """
        s = self.aircraft_params['s']
        c = self.aircraft_params['c']

        Iyy = self.aircraft_params['Iyy']
        c_m_deltae = self.aircraft_params['c_m_deltae']
        c_lift_deltae = self.aircraft_params['c_lift_deltae']
        
        #remind Austin to add the mass term in first spreadsheet
        m = self.aircraft_params['mass']
        RHO = Config.RHO
        
        G = Config.G
        Q = 0.5 * RHO * u_airspeed_ms**2

        Z_de = -c_lift_deltae * Q * s / (m*u_airspeed_ms)
        Z_deltathrust = 0.0
                
        X_de = 0.0
        X_dt = 1/(m)
        
        Z_de = -Q * c_lift_deltae * s / m
        M_de = -c_m_deltae * (Q * s * c) /(Iyy) 
        
        B = np.array([
            [X_de, X_dt],
            [Z_de, Z_deltathrust],
            [M_de, 0],
            [0  ,  0],
            [0  ,  0],
            [0  ,  0]
            ]) 
        
        #extend to 6 more rows for the lateral states
        B_lon = np.vstack((B, np.zeros((6, B.shape[1]))))
    
        return B_lon
    
    def compute_B_lat(self, velocity:float)->np.ndarray:
        """
        computes the B lateral matrix of the aircraft
        """
        Q = 0.5 * Config.RHO * velocity**2
        s = self.aircraft_params['s']
        b = self.aircraft_params['b']
        m = self.aircraft_params['mass']
        
        Ix = self.aircraft_params['Ixx']
        Iz = self.aircraft_params['Izz']
        
        c_l_da = self.aircraft_params['c_l_deltaa']
        c_y_dr = self.aircraft_params['c_y_deltar']
        
        c_l_dr = self.aircraft_params['c_l_deltar']
        c_n_da = self.aircraft_params['c_n_deltaa']
        c_n_dr = self.aircraft_params['c_n_deltar']
        
        L_da = Q * s * b * c_l_da / Ix;
        L_dr = Q * s * b * c_l_dr / Ix;
        
        N_da = Q * s * b * c_n_da / Iz;
        N_dr = Q * s * b * c_n_dr / Iz;
        
        Ydr = Q * s * c_y_dr / m;
        
        B = np.array([[0.0 , Ydr/velocity],
             [L_da, L_dr],
             [N_da, N_dr],
             [0.0 , 0.0], 
             [0.0 ,  0.0],
             [0.0 ,  0.0]])
                
        B_lat = np.vstack((np.zeros((6, B.shape[1])), B))
        
        return B_lat

    def compute_B_full(self, velocity:float)->np.ndarray:
        """
        
        """        
        B_lon = self.compute_B_lon(velocity)
        B_lat = self.compute_B_lat(velocity)
        
        #combine the two matrices
        B_full = np.hstack((B_lon, B_lat))
        
        return B_full
    
    def compute_derivatives(self, 
                            input_elevator_rad:float,
                            input_thrust:float,
                            input_aileron_rad:float,
                            input_rudder_rad:float,
                            states:np.ndarray,
                            A:np.ndarray,
                            B:np.ndarray) -> np.ndarray:
        
        thrust_scale = self.aircraft_params['mass'] * 9.81 / \
            Config.HOVER_THROTTLE
            
        input_thrust = input_thrust * thrust_scale
        
        controls = np.array([input_elevator_rad, input_thrust,
                             input_aileron_rad, input_rudder_rad])
        
        x_dot = np.matmul(A, states) + np.matmul(B, controls)
        
        return x_dot
    
    def rk45(self,
            input_elevator_rad:float,
            input_thrust:float,
            input_aileron_rad:float,
            input_rudder_rad:float,
            states:np.ndarray, A:np.ndarray,
            B:np.ndarray,
            delta_time:float) -> np.ndarray:
        """
        Simulates the aircraft using the Runge-Kutta 4th order method 
        """
        #get the current states
        current_states = states

        #compute the derivatives
        k1 = delta_time * self.compute_derivatives(input_elevator_rad,
                                        input_thrust,
                                        input_aileron_rad,
                                        input_rudder_rad,
                                        current_states, A, B)
        
        k2 = delta_time * self.compute_derivatives(input_elevator_rad,
                                        input_thrust,
                                        input_aileron_rad,
                                        input_rudder_rad,
                                        current_states + k1/2, A, B)
        
        k3 = delta_time * self.compute_derivatives(input_elevator_rad,
                                        input_thrust,
                                        input_aileron_rad,
                                        input_rudder_rad,
                                        current_states + k2/2, A, B)
        
        k4 = delta_time * self.compute_derivatives(input_elevator_rad,
                                        input_thrust,
                                        input_aileron_rad,
                                        input_rudder_rad,
                                        current_states + k3, A, B)
        
        new_states = current_states + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        return new_states


class LinearizedAircraftCasadi():
    """
    Linearized aircraft model for use with casadi
    A =  12 x 12 matrix
    B = 12 x 4 matrix
    
    states are ordered as follows:
        [u, w, q, theta, z, x, v, p, r, phi, psi, y]
        First half is longitudinal
        Second half is lateral

    """
    def __init__(self, aircraft_params:dict, A:np.ndarray,
                 B:np.ndarray) -> None:
        self.aircraft_params = aircraft_params
        #velocity trim condition for the aircraft
        self.define_states()
        self.define_controls()
        self.thrust_scale = self.aircraft_params['mass'] * 9.81 / \
            Config.HOVER_THROTTLE
        
        self.A = A
        self.B = B
        
        self.set_state_space()
        
        # self.use_own_A = False
        # self.use_own_B = False
        
        # self.compute_A()
        # self.compute_B()
    
    def define_states(self) -> None:
        #longitudinal states
        self.u = ca.MX.sym('u')
        self.w = ca.MX.sym('w')
        self.q = ca.MX.sym('q')
        self.theta = ca.MX.sym('theta')
        self.z = ca.MX.sym('z')
        self.x = ca.MX.sym('x')
        
        #lateral states
        self.v = ca.MX.sym('v')
        self.p = ca.MX.sym('p')
        self.r = ca.MX.sym('r')
        self.phi = ca.MX.sym('phi')
        self.psi = ca.MX.sym('psi')
        self.y = ca.MX.sym('y')
        
        self.states = ca.vertcat(self.u, 
                                 self.w,
                                 self.q,
                                 self.theta,
                                 self.z,
                                 self.x, 
                                 self.v,
                                 self.p,
                                 self.r,
                                 self.phi,
                                 self.psi,
                                 self.y)
        
        self.n_states = self.states.size()[0]
        
    def define_controls(self) -> None:
        self.delta_e = ca.MX.sym('delta_e')
        self.delta_t = ca.MX.sym('delta_t')
        self.delta_a = ca.MX.sym('delta_a')
        self.delta_r = ca.MX.sym('delta_r')
        
        self.controls = ca.vertcat(self.delta_e,
                                   self.delta_t,
                                   self.delta_a,
                                   self.delta_r)
        
        self.n_controls = self.controls.size()[0]
        
        
    def set_state_space(self) -> None:
        A = self.A
        B = self.B
        print("controls", self.controls)
        thrust = self.controls[1] * self.thrust_scale
        controls = ca.vertcat(self.delta_e, thrust,
                              self.delta_a, self.delta_r)
        
        Bu = ca.mtimes(B, controls)
        
        self.z_dot = ca.mtimes(A, self.states) + Bu
        
        self.f = ca.Function('dynamics', [self.states, self.controls],
                                    [self.z_dot])
        
        
    
    
    
    
