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
        beta_rad = np.arcsin(v/airspeed)
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
        u_dot = forces[0]/mass - gravity_body_frame[0] - (q*w)  + (r*v)
        v_dot = forces[1]/mass + gravity_body_frame[1] - (r*u)  + (p*w)
        w_dot = forces[2]/mass + gravity_body_frame[2] - (p*v)  + (q*u)

        u_dot = np.clip(-Config.ACCEL_LIM, Config.ACCEL_LIM, u_dot)
        v_dot = np.clip(-Config.ACCEL_LIM, Config.ACCEL_LIM, v_dot)
        w_dot = np.clip(-Config.ACCEL_LIM, Config.ACCEL_LIM, w_dot)
        

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
            constant = u_airspeed_ms*s_theta + w*c_theta
            alpha = np.arctan2(w, u_airspeed_ms)
        else:
            constant = u_airspeed_ms*s_theta
            alpha = theta_rad

        #lift coefficient 
        #C_l = (m*G / (Q*s)) * alpha
        C_l = self.lift_coeff(alpha)

        # c_drag_a = c_drag_p + np.power(c_lift_0 + c_lift_a*alpha,2) / \
        #     (np.pi * oswald * ar)
        k = 1/(np.pi) * oswald * ar
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
        
        A = np.array([
            [X_u, X_w, 0,            -G, 0],
            [Z_u, Z_w, Z_q,          0, 0],
            [M_u, M_w, M_q,           0, 0],
            [0 ,  0,   1,             0, 0],
            [-s_theta ,  -c_theta,   0, constant, 0]])
        
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
        
        self.states = ca.vertcat(self.u, 
                                 self.w, 
                                 self.q, 
                                 self.theta,
                                 self.h)
        
        self.n_states = self.states.size()[0]
    
    def define_controls(self) -> None:
        
        self.thrust_scale = self.aircraft_params['mass'] * 9.81 / \
            Config.HOVER_THROTTLE
            
        self.delta_e = ca.MX.sym('delta_e')
        self.delta_thrust = ca.MX.sym('delta_thrust')
        self.controls = ca.vertcat(self.delta_e, 
                                   self.delta_thrust)
        
        self.n_controls = self.controls.size()[0]
    
    
    def compute_A(self) -> None:

        u = self.states[0]
        w = self.states[1]
        
        airspeed = ca.sqrt(u**2 + w**2)
        
        q = self.states[2]
        theta = self.states[3]
        alpha_rad = ca.if_else(u < 1e-2, 0.0 , ca.atan2(w, u))
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
        
        c_drag_a = k * (C_l)**2          
        c_drag_u = self.aircraft_params['c_drag_u']
        
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
                
        s_theta = ca.sin(theta)
        c_theta = ca.cos(theta)

        constant = (u*s_theta) + (w*c_theta)
        
        #create a casadi 5 x 5 matrix
        A = ca.vertcat(
            ca.horzcat(X_u, X_w, X_q,  -G*c_theta, 0),
            ca.horzcat(Z_u, Z_w, Z_q, -G*s_theta, 0),
            ca.horzcat(M_u, M_w, M_q, 0, 0),
            ca.horzcat(0 ,  0,   1,             0),
            ca.horzcat(-s_theta ,  -c_theta,   0, constant, 0))
                
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
        thrust = self.controls[1] * self.thrust_scale
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
            ca.horzcat(0  ,  0))

        controls = ca.vertcat(self.delta_e, thrust)
        Bu = ca.mtimes(B, controls)  


        self.B_function = ca.Function('compute_B',
                                    [self.states, self.controls],
                                    [Bu],
                                    ['B_states', 'controls'], ['Bu'])
        
        return Bu
    

    def set_state_space(self) -> None:
        #map thrust to newtons 
        
        if self.use_own_A == False:
            A = self.A_function(self.states)
        else:
            A = self.A
            
        if self.use_own_B == False:
            Bu = self.B_function(self.states, self.controls)
        else:
            B = self.B
            #normalize thrust to newtons
            thrust = self.controls[1] * self.thrust_scale
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
        
        Y_beta = Q * s * c_y_b / m
        Y_p = Q * s * b * c_y_p / (2 * m * velocity)
        Y_r = Q * s * b * c_y_r / (2 * m * velocity)
        
        L_beta = Q * s * b * c_l_b / Ix
        L_p = Q * s * b**2 * c_l_p / (2 * Ix * velocity)
        L_r =  Q * s * b**2 * c_l_r / (2 * Ix  * velocity)

        N_beta = Q * s * b * c_n_b / Iz;
        N_p = Q * s * b**2 * c_n_p / (2 * Iz * velocity)
        N_r = Q * s * b**2 * c_n_r / (2 * Iz * velocity)
        N_dr = Q * s * b * c_n_deltar / Iz;
        
        G = Config.G
        c_theta = np.cos(theta_rad)
        
        A = [[Y_beta/velocity,  Y_p/velocity,   -(1 - (Y_r/velocity)),      (G*c_theta)/velocity],
              [L_beta,           L_p,            L_r,                           0],
              [N_beta,           N_p,            N_r,                           0],
              [0,                 1,             0,                             0]]
        
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
        
        B = [[0.0 , Ydr/velocity],
             [L_da, L_dr],
             [N_da, N_dr],
             [0.0 , 0.0]]
        
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
        
        self.states = ca.vertcat(self.v, 
                                 self.p, 
                                 self.r, 
                                 self.phi,
                                 self.psi)
        
        self.n_states = self.states.size()[0]
    
    def define_controls(self) -> None:
        self.delta_a = ca.MX.sym('delta_a')
        self.delta = ca.MX.sym('delta_r')
        self.controls = ca.vertcat(self.delta_a, 
                                   self.delta)
        
        self.n_controls = self.controls.size()[0]
        
        
        