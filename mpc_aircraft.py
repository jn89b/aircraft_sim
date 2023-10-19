import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def ca_euler_dcm_inertial_to_body(ca_phi_rad:ca.MX, 
                               ca_theta_rad:ca.MX, 
                               ca_psi_rad:ca.MX) -> ca.MX:
    """
    This computes the DCM matrix going from inertial to body frame using CasADi.
    """

    roll = ca_phi_rad
    pitch = ca_theta_rad
    yaw = ca_psi_rad

    # Create rotation matrices for each axis
    R_roll = ca.vertcat(ca.horzcat(1, 0, 0),
                        ca.horzcat(0, ca.cos(roll), -ca.sin(roll)),
                        ca.horzcat(0, ca.sin(roll), ca.cos(roll)))

    R_pitch = ca.vertcat(ca.horzcat(ca.cos(pitch), 0, ca.sin(pitch)),
                        ca.horzcat(0, 1, 0),
                        ca.horzcat(-ca.sin(pitch), 0, ca.cos(pitch)))

    R_yaw = ca.vertcat(ca.horzcat(ca.cos(yaw), -ca.sin(yaw), 0),
                    ca.horzcat(ca.sin(yaw), ca.cos(yaw), 0),
                    ca.horzcat(0, 0, 1))

    # Compute the total rotation matrix (DCM) by multiplying individual matrices
    dcm = R_yaw @ R_pitch @ R_roll
    

    return dcm


def ca_euler_dcm_body_to_inertial(ca_phi_rad: ca.MX, 
                               ca_theta_rad: ca.MX, 
                               ca_psi_rad: ca.MX) -> ca.MX:
    """
    This computes the DCM matrix going from body to inertial frame using CasADi.
    """
    # Call the function to get the DCM from inertial to body frame
    dcm_inert_to_body = ca_euler_dcm_inertial_to_body(
        ca_phi_rad,ca_theta_rad, ca_psi_rad)
    
    # Compute the DCM from body to inertial frame by taking the transpose
    dcm_body_to_inertial = dcm_inert_to_body.T
    
    return dcm_body_to_inertial

def ca_compute_B_matrix(ca_phi_rad: ca.MX, 
                        ca_theta_rad: ca.MX, 
                        ca_psi_rad: ca.MX) -> ca.MX:
    """
    Computes the B matrix for the body frame using CasADi.
    """
    # Convert input angles to CasADi MX variables
    phi = ca_phi_rad
    theta = ca_theta_rad
    psi = ca_psi_rad
    
    # Compute the B matrix elements
    B = ca.vertcat(
        ca.horzcat(ca.cos(theta), ca.sin(phi) * ca.sin(theta), ca.cos(phi) * ca.sin(theta)),
        ca.horzcat(0, ca.cos(phi) * ca.cos(theta), -ca.sin(phi) * ca.cos(theta)),
        ca.horzcat(0, ca.sin(phi), ca.cos(phi))
    )


    # Divide the matrix by cos(theta)
    B = B/ ca.cos(theta)

    return B

class AircraftCasadi():
    def __init__(self,aircraft_params:dict) -> None:
        self.define_states()
        self.define_controls()
        self.aircraft_params = aircraft_params


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
        self.delta_e = ca.MX.sym('delta_e') # Elevator
        self.delta_a = ca.MX.sym('delta_a') # Aileron
        self.delta_r = ca.MX.sym('delta_r') # Rudder
        self.delta_t = ca.MX.sym('delta_t') # Throttle

        self.controls = ca.vertcat(
            self.delta_e, self.delta_a, self.delta_r, self.delta_t)
        
        self.num_controls = self.controls.size()[0]

    def compute_aoa(self) -> ca.MX:
        # Compute the angle of attack
        return ca.atan2(self.w, self.u)

    def compute_beta(self) -> ca.MX:
        # Compute the sideslip angle
        return ca.asin(self.v / ca.sqrt(self.u**2 + self.v**2 + self.w**2))

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
        coefficient = self.aircraft_params
        alpha0 = coefficient["alpha_stall"]
        M = coefficient["mcoeff"]
        c_lift_0 = coefficient["c_lift_0"]
        c_lift_a0 = coefficient["c_lift_a"]

        c_lift_a = c_lift_0 + c_lift_a0 * ca_alpha

        return c_lift_a
    
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
        
        moments = ca.vertcat(la, ma, na)
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

        #check if airspeed is 0
        # threshold = 1e-2
        # is_airspeed_less_than = airspeed < threshold

        # check_airspeed_fn = ca.Function('check_airspeed_fn', 
        #                                 [airspeed], [is_airspeed_less_than])

        # if check_airspeed_fn(airspeed):
        #     f_ax_b = 0
        #     f_ay_b = 0
        #     f_az_b = 0
        # else:
        # Define the equations
        f_ax_b = qbar * (c_x_a + c_x_q * c*q / (2 * airspeed) - c_drag_deltae * ca.cos(alpha) * ca.fabs(self.delta_e) +
                        c_lift_deltae * ca.sin(alpha) * delta_e)
        f_ay_b = qbar * (c_y_0 + c_y_b * beta + c_y_p * b * p / (2 * airspeed) + c_y_r * b * r / (2 * airspeed) +
                        c_y_deltaa * delta_a + c_y_deltar * delta_r)
        f_az_b = qbar * (c_z_a + c_z_q * c*q / (2 * airspeed) - c_drag_deltae * ca.sin(alpha) * ca.fabs(self.delta_e) -
                        c_lift_deltae * ca.cos(alpha) * delta_e)

        # Define the total force in the body frame
        f_total_body = ca.vertcat(f_ax_b + delta_t, f_ay_b, f_az_b)                    
        
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
        # Define the state-space equations
        forces = self.compute_forces() # returns fx, fy, fz body frame
        moments = self.compute_moments() #returns la, ma, na

        body_acc = self.compute_ang_acc(moments)
        
        #multiply by dt and you get the the new angular velocities
        self.p_dot = body_acc[0] 
        self.q_dot = body_acc[1]
        self.r_dot = body_acc[2]

        B = ca_compute_B_matrix(self.phi, self.theta, self.psi)
        
        #dot product of B and angular velocities
        #update new attitudes
        #create a row vector of angular velocities
        attitude_rates = ca.vertcat(self.p, self.q, self.r)
        attitudes_dot = ca.mtimes(B, attitude_rates)
        self.phi_dot = attitudes_dot[0]
        self.theta_dot = attitudes_dot[1]
        self.psi_dot = attitudes_dot[2]

        #from forces need to divide by mass and gravity body frame
        g = 9.81 
        gravity_body_frame = ca.vertcat(
            g * ca.sin(self.theta),
            g * ca.sin(self.phi) * ca.cos(self.theta),
            g * ca.cos(self.phi) * ca.cos(self.theta)
        )
        mass = self.aircraft_params["mass"]
        
        acc_x_bf = (forces[0] / mass) - \
            gravity_body_frame[0] - (self.q*self.w) + (self.r*self.v)
        
        acc_y_bf = (forces[1] / mass) + \
            gravity_body_frame[1] - (self.r*self.u) + (self.p*self.w)
        
        acc_z_bf = (forces[2] / mass) + \
            gravity_body_frame[2] - (self.p*self.v) + (self.q*self.u)

        #update new velocities
        self.u_dot = acc_x_bf
        self.v_dot = acc_y_bf
        self.w_dot = acc_z_bf

        #update new positions
        dcm = ca_euler_dcm_body_to_inertial(
            self.phi, self.theta, self.psi)

        # get world frame velocities
        aircraft_vel_bf = ca.vertcat(self.u, self.v, self.w)
        aircraft_vel_wf = ca.mtimes(dcm, aircraft_vel_bf)
        self.x_dot = aircraft_vel_wf[0]
        self.y_dot = aircraft_vel_wf[1]
        self.z_dot = aircraft_vel_wf[2]

        self.z_dot = ca.vertcat(
            self.x_dot, self.y_dot, self.z_dot, 
            self.u_dot, self.v_dot, self.w_dot, 
            self.phi_dot, self.theta_dot, self.psi_dot, 
            self.p_dot, self.q_dot, self.r_dot)
        
        # self.z_dot = ca.reshape(self.z_dot, (self.num_states, 1))

        #right hand side of the state space equation
        self.state_space = ca.Function('state_space', 
                                       [self.states, self.controls], 
                                       [self.z_dot])
    
        return self.state_space 
    
def get_airplane_params(df:pd.DataFrame) -> dict:
    airplane_params = {}
    for index, row in df.iterrows():
        airplane_params[row["var_name"]] = row["var_val"]

    airplane_params["mass"] = 10 # kg
    
    return airplane_params

if __name__=="__main__":

    df = pd.read_csv("SIM_Plane_h_vals.csv")
    airplane_params = get_airplane_params(df)
    aircraft = AircraftCasadi(airplane_params)

    aircraft_ss = aircraft.set_state_space()
    print(aircraft_ss)


    init_x = 0
    init_y = 0
    init_z = 0
    init_u = 0
    init_v = 0
    init_w = 0
    init_phi = 0
    init_theta = 0
    init_psi = 0
    init_p = 0
    init_q = 0
    init_r = 0

    init_el = 0
    init_al = 0
    init_rud = 0
    init_throttle = 100


    goal_x = 10
    goal_y = 10
    goal_z = 10

    # Optimal control problem
    opti = ca.Opti()
    dt = 0.01 # Time step
    N = 10
    t_init = 0 # Initial time

    # Define the states over the optimization problem
    X = opti.variable(aircraft.num_states, N+1) # State vector
    U = opti.variable(aircraft.num_controls, N) # Control vector
    
    x0 = opti.parameter(aircraft.num_states, 1) # Initial state
    xF = opti.parameter(aircraft.num_states, 1) # Final state
    
    u0 = opti.parameter(aircraft.num_controls, 1) # Initial control input

    # set initial value
    opti.set_value(x0, np.array([init_x, init_y, init_z, 
                                 init_u, init_v, init_w, 
                                 init_phi, init_theta, init_psi, 
                                 init_p, init_q, init_r]))
    
    opti.set_value(xF, np.array([goal_x, goal_y, goal_z, 
                                 0, 0, 0, 
                                 0, 0, 0, 
                                 0, 0, 0]))
    opti.set_value(u0, np.array([init_el, init_al, init_rud, init_throttle]))

    # set initial and terminal constraints
    opti.subject_to(X[:,0] == x0)
    opti.subject_to(X[:,-1] == xF)
    # opti.subject_to(U[:,0] == u0)
    

    # set constraints to dynamics
    for k in range(N):
        states = X[:,k]
        controls = U[:,k]
        state_next = X[:,k+1]

        k1 = aircraft_ss(states, controls)
        k2 = aircraft_ss(states + dt/2 * k1, controls)
        k3 = aircraft_ss(states + dt/2 * k2, controls)
        k4 = aircraft_ss(states + dt * k3, controls)
        state_next_RK4 = states + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        opti.subject_to(state_next == state_next_RK4)

    # set constraints to control inputs
    max_control_surface = np.deg2rad(25)
    min_control_surface = np.deg2rad(-25)
    min_throttle = 50 # Newtons
    max_throttle = 200 # Newtons

    #aileron
    # opti.subject_to(opti.bounded(min_control_surface, U[0,:], max_control_surface))
    # #elevator
    # opti.subject_to(opti.bounded(min_control_surface, U[1,:], max_control_surface))
    # #ruder
    # opti.subject_to(opti.bounded(min_control_surface, U[2,:], max_control_surface))
    # #throttle
    # opti.subject_to(opti.bounded(min_throttle, U[3,:], max_throttle))

    # set cost function 
    # minimize the sum of the squares of the control inputs
    opti.minimize(ca.sumsqr(U))

    # solve the optimization problem
    opts = {
        'ipopt': {
            'max_iter': 5000,
            'print_level': 2,
            'acceptable_tol': 1e-2,
            'acceptable_obj_change_tol': 1e-2,
        },
        'print_time': 1
    }
    opti.solver('ipopt', opts)#, {'ipopt': {'print_level': 0}})


    #Initial Trajectory    
    sol = opti.solve()    
    print(sol.value(X))
    #debugging
    opti.debug.value(X)



