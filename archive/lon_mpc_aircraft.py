import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

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


class LonAircraftCasadi():
    def __init__(self, aircraft_params:dict) -> None:
        self.aircraft_params = aircraft_params

        self.define_states()
        self.define_controls()
        self.compute_forces()
        self.compute_moments()

    def define_states(self):
        self.u = ca.MX.sym('u')
        self.w = ca.MX.sym('w')
        self.theta = ca.MX.sym('theta')
        self.q = ca.MX.sym('q')

        self.states = ca.vertcat(self.u, self.w, self.theta, self.q)
        self.num_states = self.states.shape[0]

    def define_controls(self) -> None:
        self.de = ca.MX.sym('de')
        self.delta_t = ca.MX.sym('delta_t')
        self.controls = ca.vertcat(
            self.de, self.delta_t)
        self.num_controls = self.controls.shape[0]

    def compute_aoa(self) -> ca.MX:
        # Compute the angle of attack
        airspeed = ca.sqrt(self.u**2 + self.w**2)

        return ca.atan2(self.w, self.u)   
        #return ca.asin(self.w / airspeed)
    
    def lift_coeff(self, ca_alpha)-> ca.MX:
        coefficient = self.aircraft_params
        alpha0 = coefficient["alpha_stall"]
        M = coefficient["mcoeff"]
        c_lift_0 = coefficient["c_lift_0"]
        c_lift_a0 = coefficient["c_lift_a"]
        # alpha = alpha0
        alpha = ca_alpha

        # sigmoid = (1 + np.exp(-M * (alpha - alpha0)) + \
        #            np.exp(M * (alpha + alpha0))) / (1 + math.exp(-M * (alpha - alpha0))) \
        #             / (1 + math.exp(M * (alpha + alpha0)))
        
        #use casadi to compute the sigmoid function
        sigmoid = (1 + ca.exp(-M * (alpha - alpha0)) + \
                     ca.exp(M * (alpha + alpha0))) / (1 + ca.exp(-M * (alpha - alpha0))) \
                        / (1 + ca.exp(M * (alpha + alpha0)))
        

        linear = (1.0 - sigmoid) * (c_lift_0 + c_lift_a0 * alpha)  # Lift at small AoA
        #flat_plate = sigmoid * (2 * math.copysign(1, alpha) * math.pow(math.sin(alpha), 2) * math.cos(alpha))  # Lift beyond stall

        #use casadi to compute the flat plate function
        flat_plate = sigmoid * (2 * ca.sign(alpha) * ca.sin(alpha)**2 * ca.cos(alpha))

        result = linear + flat_plate
        return result
        #return c_lift_0 + c_lift_a0 * ca_alpha
    
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
    
    def compute_moments(self) -> ca.MX:
        alpha = self.compute_aoa()
        # beta = self.compute_beta()
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

        # airspeed = ca.sqrt(self.u**2 + self.v**2 + self.w**2)
        airspeed = ca.sqrt(self.u**2 + self.w**2)
        effective_airspeed = airspeed
        qbar = 0.5 * rho * effective_airspeed**2

        # input_aileron = self.delta_a
        # input_rudder = self.delta_r
        # input_elevator = self.delta_e
        input_elevator = self.de

        #check if airspeed is 0

        # la = qbar * b * (c_l_0 + c_l_b * beta +
        #                 c_l_p * b * self.p / (2 * effective_airspeed) +
        #                 c_l_r * b * self.r / (2 * effective_airspeed) +
        #                 c_l_deltaa * input_aileron +
        #                 c_l_deltar * input_rudder)

        # ma = qbar * c * (c_m_0 + c_m_a * alpha +
        #                     c_m_q * c * self.q / (2 * effective_airspeed) +
        #                     c_m_deltae * input_elevator)

        #simplify the moment equations
        ma = qbar * c * (c_m_0 + c_m_a * alpha +
                            c_m_q * c * self.q / (2 * effective_airspeed) +
                            c_m_deltae * input_elevator)

        # na = qbar * b * (c_n_0 + c_n_b * beta + c_n_p * b * self.p / (2 * effective_airspeed) +
        #             c_n_r * b * self.r / (2 * effective_airspeed) +
        #             c_n_deltaa * input_aileron +
        #             c_n_deltar * input_rudder)
        
        #moments = ca.vertcat(la, ma, na)
        

        ma = ca.if_else(airspeed < 1e-2, 0, ma)
    
        moments = ca.vertcat(ma)


        self.moment_function = ca.Function('compute_moments',
                                        [self.states, self.controls],
                                        [moments],
                                        ['states', 'controls'], ['moments'])
        
        return moments
    
    def compute_forces(self) -> ca.MX:
        alpha = self.compute_aoa()
        # beta = self.compute_beta()
        # airspeed = ca.sqrt(self.u**2 + self.v**2 + self.w**2)
        airspeed = ca.sqrt(ca.fmax(self.u**2 + self.w**2, 0.0))
        # p = self.p
        q = self.q
        # r = self.r
        delta_e = self.de
        # delta_a = self.delta_a
        # delta_r = self.delta_r
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
        f_ax_b = qbar * (c_x_a + c_x_q * c*q / (2 * airspeed) - \
                         c_drag_deltae * ca.cos(alpha) * delta_e +
                        c_lift_deltae * ca.sin(alpha) * delta_e)
        
        # f_ay_b = qbar * (c_y_0 + c_y_b * beta + c_y_p * b * p / (2 * airspeed) + c_y_r * b * r / (2 * airspeed) +
        #                 c_y_deltaa * delta_a + c_y_deltar * delta_r)
        
        # control_delta_a = ca.if_else(airspeed < 1e-2, 0, self.de)

        f_az_b = qbar*(c_z_a + c_z_q*c*q/(2*airspeed) - \
                c_drag_deltae*np.sin(alpha)*ca.fabs(delta_e) - \
                c_lift_deltae*np.cos(alpha)*delta_e)


        # Set forces to zero when airspeed is zero
        f_ax_b = ca.if_else(airspeed < 1e-2, 0.0, f_ax_b)
        f_az_b = ca.if_else(airspeed < 1e-2, 0.0, f_az_b)

        # f_az_b = -10
        # Define the total force in the body frame
        f_total_body = ca.vertcat(f_ax_b + delta_t, f_az_b)                    
        
        self.force_function = ca.Function('compute_forces', 
                                        [self.states, self.controls], 
                                        [f_total_body],
                                        ['states', 'controls'], ['f_total_body'])   

        return f_total_body
    

    def set_state_space(self) -> ca.Function:
        Ixx = self.aircraft_params["Ixx"]
        Iyy = self.aircraft_params["Iyy"]
        Izz = self.aircraft_params["Izz"]

        moments = self.moment_function(self.states, self.controls)
        forces = self.force_function(self.states, self.controls)

        # moments[0] = 10
        # forces = ca.vertcat(15, 20)

        print("moments", moments)
        print("forces", forces)

        second_part = (Izz - Iyy) * self.q

        g = 9.81
        mass = self.aircraft_params["mass"]

        self.u_dot = (forces[0] / mass) - g * ca.sin(self.theta) #- self.q 
        self.w_dot = (forces[1] / mass) + g * ca.cos(self.theta) #- self.u
        
        self.q_dot = (moments[0] + second_part) / Ixx
        self.theta_dot = (self.q)

        self.z_dot = ca.vertcat(self.u_dot, self.w_dot, self.theta_dot, self.q_dot)

        self.f = ca.Function('f', [self.states, self.controls], [self.z_dot], 
                             ['states', 'controls'], 
                             ['z_dot'])

        return self.f

def get_airplane_params(df:pd.DataFrame) -> dict:
    airplane_params = {}
    for index, row in df.iterrows():
        airplane_params[row["var_name"]] = row["var_val"]

    airplane_params["mass"] = 10 # kg
    
    return airplane_params

if __name__=="__main__":

    df = pd.read_csv("SIM_Plane_h_vals.csv")
    airplane_params = get_airplane_params(df)
    aircraft = LonAircraftCasadi(airplane_params)
    f = aircraft.set_state_space()

    init_u = 25
    init_w = 0
    init_theta = 0
    init_q = 0

    init_state = np.array([init_u, init_w, init_theta, init_q])

    init_de = np.deg2rad(0)
    init_delta_t = 0

    init_control = np.array([init_de, init_delta_t])

    final_u = 15
    final_w = 0
    final_theta = np.deg2rad(5)
    final_q = 0 
    final_state = np.array([final_u, final_w, final_theta, final_q])

    # Optimal control problem
    opti = ca.Opti()
    dt = 0.05 # Time step
    N = 15
    t_init = 0 # Initial time

    # Define the states over the optimization problem
    X = opti.variable(aircraft.num_states, N+1) # State vector
    U = opti.variable(aircraft.num_controls, N) # Control vector
    
    x0 = opti.parameter(aircraft.num_states, 1) # Initial state
    xF = opti.parameter(aircraft.num_states, 1) # Final state
    
    u0 = opti.parameter(aircraft.num_controls, 1) # Initial control input
    
    # set initial value
    opti.set_value(x0, np.array([
                                 #init_x, init_y, init_z, 
                                 init_u, 
                                 init_w,
                                 init_theta,  
                                 init_q]))
    
    opti.set_value(u0, init_control)
    
    opti.set_value(xF, final_state) 
    opti.subject_to(X[:,0]==x0) # Use initial state

    for k in range(N):
        # Integrate till the end of the interval
        #use runge kutta 4 integration method
        #set control inputs 
        opti.subject_to(U[:,k]==u0) # Use initial control input

        k1 = f(X[:,k], U[:,k])
        k2 = f(X[:,k] + dt/2 * k1, U[:,k])
        k3 = f(X[:,k] + dt/2 * k2, U[:,k])
        k4 = f(X[:,k] + dt * k3, U[:,k])
        x_next = X[:,k] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        opti.subject_to(X[:,k+1]==x_next) # Use initial state
        #use eulers
        #opti.subject_to(X[:,k+1] == X[:,k] + dt * f(X[:,k], U[:,k]))

    # Define the cost function
    # Q = np.diag([1, 1, 1, 1])
    # R = np.diag([1, 1])
    # cost = 0
    # for k in range(N):
    #     cost += ca.mtimes([(X[:,k] - xF).T, Q, (X[:,k] - xF)]) + ca.mtimes([U[:,k].T, R, U[:,k]])

    # opti.minimize(cost)

    # opti.minimize(ca.sumsqr(U))

    # solve the optimization problem
    opts = {
        'ipopt': {
            'max_iter': 5000,
            'print_level': 2,
            'acceptable_tol': 1e-2,
            'acceptable_obj_change_tol': 1e-2,
            'hessian_approximation': 'limited-memory',  # Set Hessian approximation method here
        },
        'print_time': 2
    }
    opti.solver('ipopt', opts)#, {'ipopt': {'print_level': 0}})
    
    #Initial Trajectory    
    sol = opti.solve()    
    print(sol.value(X))
    #debugging
    opti.debug.value(X)
    
    t_vec = np.linspace(t_init, t_init + N*dt, N+1)

    # Plot the results
    fig, axs = plt.subplots(4, 1, figsize=(10, 10))
    axs[0].plot(t_vec,sol.value(X[0,:]), label='u')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('u (m/s)')
    axs[0].legend()
    axs[1].plot(t_vec,sol.value(X[1,:]), label='w')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('w (m/s)')
    axs[1].legend()

    axs[2].plot(t_vec,np.rad2deg(sol.value(X[2,:])), label='theta')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('theta (deg)')
    axs[2].legend()
    axs[3].plot(t_vec,np.rad2deg(sol.value(X[3,:])), label='q')
    axs[3].set_xlabel('Time')
    axs[3].set_ylabel('q (deg/s)')
    axs[3].legend()


    #plot the control inputs
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(t_vec[:-1],np.rad2deg(sol.value(U[0,:])), label='de')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('de (deg)')
    axs[0].legend()
    axs[1].plot(t_vec[:-1],sol.value(U[1,:]), label='delta_t')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('delta_t')
    axs[1].legend()
    
    plt.show()





