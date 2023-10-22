import numpy as np
import math
import random
import casadi as ca

from src.aircraft.Aircraft import AircraftInfo

from src.math_lib.VectorOperations import euler_dcm_inertial_to_body, \
    compute_B_matrix, euler_dcm_body_to_inertial, ca_compute_B_matrix    
    
from src.math_lib.VectorOperations import ca_euler_dcm_body_to_inertial, \
    ca_euler_dcm_inertial_to_body
 

class AircraftDynamics():
    def __init__(self, aircraft:AircraftInfo) -> None:
        self.aircraft = aircraft

    def compute_aoa(self, u:float, w:float) -> float:
        """
        Computes the angle of attack
        """
        #check divide by zero 
        # if u == 0:
        #     return 0.0
        
        #compute the angle of attack
        return np.arctan2(w, u)

    
    def compute_beta(self, u:float, v:float) -> float:
        """
        Computes the sideslip angle
        """
        #check divide by zero 
        # if u == 0:
        #     return 0.0
        
        return np.arctan2(v, u)
    
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
        beta = self.compute_beta(u,v)

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
            
        la += CGOffset[1] * force[2] - CGOffset[2] * force[1]
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
        beta_rad = self.compute_beta(u,v)

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
            
        f_total_body = np.array([f_ax_b+input_thrust, 
                                 f_ay_b, 
                                 f_az_b])

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
        
        phi = states[6]
        theta = states[7]
        psi = states[8]

        B = compute_B_matrix(phi, theta, psi)
        
        # compute angular velocities
        current_ang_velocities = states[9:12]
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
        u_dot = forces[0]/mass - gravity_body_frame[0] - (q*w)  + (r*v)
        v_dot = forces[1]/mass + gravity_body_frame[1] - (r*u)  + (p*w)
        w_dot = forces[2]/mass + gravity_body_frame[2] - (p*v)  + (q*u)


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

    def compute_aoa(self) -> ca.MX:
        # Compute the angle of attack
        # check divide by zero
        alpha = ca.if_else(self.u < 1e-2, 0.0 , ca.atan2(self.w, self.u))
        return alpha

    def compute_beta(self) -> ca.MX:
        # Compute the sideslip angle
        # check divide by zero

        beta = ca.if_else(self.u < 1e-2, 0.0, ca.atan2(self.v, self.u))
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
    
