import numpy as np
import math 
from src.VectorOperations import euler_dcm_inertial_to_body, euler_dcm_body_to_inertial
from src.VectorOperations import compute_B_matrix

class Vector3D():
    def __init__(self) -> None:
        pass

class Aircraft():
    def __init__(self, aircraft_params:dict) -> None:
        self.aircraft_params =  aircraft_params
        
        self.velocity_bf = [0, 0, 0] #u,v,w
        self.acc_bf = [0, 0, 0] #u_dot, v_dot, w_dot

        self.angular_velocity_bf = [0, 0, 0] #p,q,r
        self.angular_acc_bf = [0, 0, 0]  #p_dot, q_dot, r_dot or  l,m,n
        self.attitudes = [0, 0, 0] #phi, theta, psi





class AircraftSim():
    def __init__(self, aircraft:Aircraft) -> None:
        self.aircraft = aircraft

    def compute_aoa(self):
        """computes the angle of attack in radians"""
        alpha_rad = np.arctan2(self.aircraft.velocity_bf[2], self.aircraft.velocity_bf[0])

        return alpha_rad
    def compute_beta(self):
        """computes the sideslip angle in radians"""
        return np.arctan2(self.aircraft.velocity_bf[1], self.aircraft.velocity_bf[0])

    ### advancing rotational quantities
    def compute_moments(self)-> np.ndarray:
        """
        Returns the L,M,N moments using Rotational Dynamic Equations

        Use Austin/Ardupilot
        """
        
        p = self.aircraft.angular_velocity_bf[0]
        q = self.aircraft.angular_velocity_bf[1]
        r = self.aircraft.angular_velocity_bf[2]
        
        Ixx = self.aircraft.aircraft_params['Ixx']
        Iyy = self.aircraft.aircraft_params['Iyy']
        Izz = self.aircraft.aircraft_params['Izz']

        L = p * Ixx + q * r * (Izz - Iyy) - (r + p * q)
        M = q * Iyy - p * r * (Izz - Ixx) + (p ** 2 + r ** 2)
        N = r * Izz + p * q * (Iyy - Ixx) + (q * r - p) 
        
        moments = np.array([L, M, N]) 
        return moments
    
    def update_angular_acc(self, moments:np.ndarray) -> None:
        """
        This updates the aircraft p_dot, q_dot, r_dot
        """
        self.aircraft.angular_acc_bf = moment

    def update_angular_velocity(self, moments:np.ndarray, 
                                delta_time:float)->None:
        """
        Updates the angular velocity for the body frame : p,q,r
        Refer to this 
        https://academicflight.com/articles/aircraft-attitude-and-euler-angles/

        This 
        """
        # compute angular acceleration using rotational dynamic equations
        self.aircraft.angular_velocity_bf += moments * delta_time

    def update_attitude(self, delta_time:float) ->None:
        """
        Update the attitudes of the aircraft using the angular velocity 
        """
        B = compute_B_matrix(self.aircraft.attitudes[0],
                                self.aircraft.attitudes[1],
                                self.aircraft.attitudes[2])
        self.aircraft.attitudes += B.dot(self.aircraft.angular_velocity_bf)*delta_time

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

        result = linear + flat_plate
        return result

    def drag_coeff(self, alpha) -> float:
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
        c_drag_a = c_drag_p + pow(c_lift_0 + c_lift_a0 * alpha, 2) / (math.pi * oswald * ar)

        return c_drag_a

    def compute_forces(self, input_aileron_rad:float,
                       input_elevator_rad:float,
                       input_rudder_rad:float,
                       input_thrust:float) -> np.ndarray:
        """
        Computes the forces in the body frame
        """
        # from https://academicflight.com/articles/aircraft-attitude-and-euler-angles/
        alpha_rad = self.compute_aoa()
        beta_rad = self.compute_beta()

        coefficient = self.aircraft.aircraft_params
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
        rho = 1.225 #kg/m^3

        # get lift and drag alpha coefficients 
        c_lift_a = self.lift_coeff(alpha_rad)
        c_drag_a = self.drag_coeff(alpha_rad)

        # coefficients to the body frame
        c_x_a = -c_drag_a*np.cos(alpha_rad) + c_lift_a*np.sin(alpha_rad)
        c_x_q = -c_drag_q*np.cos(alpha_rad) + c_lift_q*np.sin(alpha_rad)
        c_z_a = -c_drag_a*np.sin(alpha_rad) - c_lift_a*np.cos(alpha_rad)
        c_z_q = -c_drag_q*np.sin(alpha_rad) - c_lift_q*np.cos(alpha_rad)

        #compute angular rates
        p = self.aircraft.angular_velocity_bf[0]
        q = self.aircraft.angular_velocity_bf[1]
        r = self.aircraft.angular_velocity_bf[2]

        #compute the aerodynamic forces
        qbar = 0.5*rho*self.aircraft.airspeed**2

        #check close to zero
        airspeed = self.aircraft.airspeed
        if airspeed == 0:
            ax = 0 
            ay = 0
            az = 0
        else:
            ax = qbar*(c_x_a + c_x_q*c*q/(2*airspeed) - \
                       c_drag_deltae*np.cos(alpha_rad)*abs(input_elevator_rad) + \
                        c_lift_deltae*np.sin(alpha_rad)*input_elevator_rad)
            ay = qbar*(c_y_0 + c_y_b*beta_rad + c_y_p*b*p/(2*airspeed) + \
                       c_y_r*b*r/(2*airspeed) + c_y_deltaa*input_aileron_rad + c_y_deltar*input_rudder_rad)
            az = qbar*(c_z_a + c_z_q*c*q/(2*airspeed) - \
                 c_drag_deltae*np.sin(alpha_rad)*abs(input_elevator_rad) - \
                    c_lift_deltae*np.cos(alpha_rad)*input_elevator_rad)
            
        forces = np.array([ax, ay, az])
        return forces        

# Update angular velocity 

