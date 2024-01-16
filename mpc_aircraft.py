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

    cos_phi = ca.cos(roll)
    sin_phi = ca.sin(roll)
    cos_theta = ca.cos(pitch)
    sin_theta = ca.sin(pitch)
    cos_psi = ca.cos(yaw)
    sin_psi = ca.sin(yaw)

    dcm = ca.vertcat(
        ca.horzcat(cos_theta * cos_psi, cos_theta * sin_psi, -sin_theta),
        ca.horzcat(sin_phi * sin_theta * cos_psi - cos_phi * sin_psi, 
                    sin_phi * sin_theta * sin_psi + cos_phi * cos_psi, 
                    sin_phi * cos_theta),
        ca.horzcat(cos_phi * sin_theta * cos_psi + sin_phi * sin_psi, 
                    cos_phi * sin_theta * sin_psi - sin_phi * cos_psi, 
                    cos_phi * cos_theta)
    )
    
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

    f = aircraft.set_state_space()

    init_x = 10
    init_y = 60
    init_z = 5
    init_u = 25
    init_v = 0
    init_w = 0
    init_phi = 0
    init_theta = np.deg2rad(-5)
    init_psi = np.deg2rad(38)
    init_p = 0
    init_q = 0
    init_r = 0

    init_el = np.deg2rad(0)
    init_al = np.deg2rad(0)
    init_rud = 0
    init_throttle = 30

    goal_x = 95
    goal_y = 105
    goal_z = -7
    goal_phi = np.deg2rad(-21)
    goal_theta = np.deg2rad(0)
    goal_psi = np.deg2rad(45)

    states = np.array([
        init_x, init_y, init_z,
        init_u, init_v, init_w,
        init_phi, init_theta, init_psi,
        init_p, init_q, init_r
    ])
    controls = np.array([init_al, init_el, init_rud, init_throttle])

    #set max attitude
    max_phi = np.deg2rad(45)
    min_phi = np.deg2rad(-45)
    max_theta = np.deg2rad(25)
    min_theta = np.deg2rad(-25)
    min_vel = 15
    max_vel = 30

    #set the control surface limits
    max_control_surface = np.deg2rad(25)
    min_control_surface = np.deg2rad(-25)
    min_throttle = 20 # Newtons
    max_throttle = 200 # Newtons

    # Optimal control problem
    opti = ca.Opti()
    dt = 1/100 # Time step
    N = 100
    t_init = 0 # Initial time

    # Define the states over the optimization problem
    X = opti.variable(aircraft.num_states, N+1) # State vector
    U = opti.variable(aircraft.num_controls, N) # Control vector
    
    x0 = opti.parameter(aircraft.num_states, 1) # Initial state
    xF = opti.parameter(aircraft.num_states, 1) # Final state
    
    u0 = opti.parameter(aircraft.num_controls, 1) # Initial control input

    # set initial value
    opti.set_value(x0, np.array([
                                 init_x, init_y, init_z, 
                                 init_u, init_v, init_w, 
                                 init_phi, init_theta, init_psi, 
                                 init_p, init_q, init_r]))
    
    opti.set_value(xF, np.array([
                                 goal_x, goal_y, goal_z, 
                                 25, 0, 0, 
                                 0, 0, 0, 
                                 0, 0, 0]))
    opti.set_value(u0, controls)

    # set initial and terminal constraints
    opti.subject_to(X[:,0] == x0)
    # opti.subject_to(X[:,-1] == xF)
    opti.subject_to(U[:,0] == u0)

    # set constraints
    #set the max and min values for the states
    opti.subject_to(opti.bounded(min_vel, X[3,:], max_vel))
    opti.subject_to(opti.bounded(min_phi, X[6,:], max_phi))
    opti.subject_to(opti.bounded(min_theta, X[7,:], max_theta))
    
    #aileron
    opti.subject_to(opti.bounded(min_control_surface, U[0,:], max_control_surface))
    #elevator
    opti.subject_to(opti.bounded(min_control_surface, U[1,:], max_control_surface))
    #ruder
    opti.subject_to(opti.bounded(min_control_surface, U[2,:], max_control_surface))
    #throttle
    opti.subject_to(opti.bounded(min_throttle, U[3,:], max_throttle))

    # set cost function 
    # minimize the error between the final state and the goal state
    #set weights for the cost function as array
    weights = np.array([1.0, 1.0, 1.0, 
                        0, 0, 0,
                        0, 0, 0.5, 
                        0, 0, 0])
    
    weights_controls = np.array([0.5, 0.5, 0.5, 0.5])

    #set the cost function to minimize error and the control inputs
    # state_error = weights*ca.sumsqr((X[:, -1] - xF))
    cost_fn = ca.sumsqr(weights * (X[:, -1] - xF) + ca.sumsqr(weights_controls*U**2))
    opti.minimize(cost_fn) 


    for k in range(N):
        
        k1 = f(X[:,k], U[:,k])
        k2 = f(X[:,k] + dt/2 * k1, U[:,k])
        k3 = f(X[:,k] + dt/2 * k2, U[:,k])
        k4 = f(X[:,k] + dt * k3, U[:,k])
        x_next = X[:,k] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        opti.subject_to(X[:,k+1]==x_next) # Use initial state

        # use eulers
        # opti.subject_to(X[:,k+1] == X[:,k] + (dt * f(X[:,k], U[:,k])))

    # solve the optimization problem
    opts = {
        'ipopt': {
            'max_iter': 2000,
            'print_level': 2,
            'acceptable_tol': 1e-2,
            'acceptable_obj_change_tol': 1e-2,
            'hessian_approximation': 'limited-memory',  # Set Hessian approximation method here
        },
        'print_time': 1
    }


    opti.solver('ipopt', opts)#, {'ipopt': {'print_level': 0}})

    #Initial Trajectory    
    sol = opti.solve()    
    print("Opti solver", opti)
    print(sol.value(X))
    #debugging
    opti.debug.value(X)

    #plot everything
    
    # plot position in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(sol.value(X[0,:]), sol.value(X[1,:]), -sol.value(X[2,:]) , '-o')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')

    #plot starting and ending points
    ax.scatter(sol.value(X[0,0]), sol.value(X[1,0]), 
               -sol.value(X[2,0]), c='r', marker='o', label='start')
    # ax.scatter(sol.value(X[0,-1]), sol.value(X[1,-1]), 
    #            -sol.value(X[2,-1]), c='c', marker='o', label='end')

    #plot the goal
    print("goal", goal_x, goal_y, goal_z)
    ax.scatter(goal_x, goal_y, -goal_z, c='g', marker='x', label='goal')

    #set axis equal
    max_range = np.array([sol.value(X[0,:]).max()-sol.value(X[0,:]).min(), sol.value(X[1,:]).max()-sol.value(X[1,:]).min(), sol.value(X[2,:]).max()-sol.value(X[2,:]).min()]).max() / 2.0
    mean_x = sol.value(X[0,:]).mean()
    mean_y = sol.value(X[1,:]).mean()
    mean_z = sol.value(X[2,:]).mean()
    max_z_range = np.array([sol.value(X[2,:]).max()-sol.value(X[2,:]).min()]).max() / 2.0
    # ax.set_xlim(mean_x - max_range, mean_x + max_range)
    # ax.set_ylim(mean_y - max_range, mean_y + max_range)
    # ax.set_zlim(mean_z - max_z_range, mean_z + 30)
    ax.legend()

    t_vec = np.linspace(t_init, t_init + N*dt, N)
    #plot control inputs in subplots
    fig, axs = plt.subplots(4,1)
    axs[0].plot(t_vec, np.rad2deg(sol.value(U[0,:])))
    axs[0].set_ylabel('Aileron [deg]')
    axs[1].plot(t_vec, np.rad2deg(sol.value(U[1,:])))
    axs[1].set_ylabel('Elevator [deg]')
    axs[2].plot(t_vec, np.rad2deg(sol.value(U[2,:])))
    axs[2].set_ylabel('Rudder [deg]')
    axs[3].plot(t_vec,sol.value(U[3,:]))
    axs[3].set_ylabel('Throttle [N]')

    # plot the attitudes
    t_vec = np.linspace(t_init, t_init + N*dt, N+1)
    fig, axs = plt.subplots(3,1)
    axs[0].plot(t_vec, np.rad2deg(sol.value(X[6,:])))
    axs[0].set_ylabel('Roll [deg]')
    axs[1].plot(t_vec, np.rad2deg(sol.value(X[7,:])))
    axs[1].set_ylabel('Pitch [deg]')
    axs[2].plot(t_vec, np.rad2deg(sol.value(X[8,:])))
    axs[2].set_ylabel('Yaw [deg]')
    axs[2].set_xlabel('Time [s]')

    #plot the velocities
    fig, axs = plt.subplots(3,1)
    axs[0].plot(t_vec, sol.value(X[3,:]))
    axs[0].set_ylabel('u [m/s]')
    axs[1].plot(t_vec, sol.value(X[4,:]))
    axs[1].set_ylabel('v [m/s]')
    axs[2].plot(t_vec, sol.value(X[5,:]))
    axs[2].set_ylabel('w [m/s]')
    axs[2].set_xlabel('Time [s]')
    
    #plot control inputs in subplots
    fig, axs = plt.subplots(3,1)
    axs[0].plot(t_vec, np.rad2deg(sol.value(X[9,:])))
    axs[0].set_ylabel('p [deg/s]')
    axs[1].plot(t_vec, np.rad2deg(sol.value(X[10,:])))
    axs[1].set_ylabel('q [deg/s]')
    axs[2].plot(t_vec, np.rad2deg(sol.value(X[11,:])))
    axs[2].set_ylabel('r [deg/s]')


    plt.show()


