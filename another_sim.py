import numpy as np
import math 
import pandas as pd
from src.VectorOperations import euler_dcm_inertial_to_body, euler_dcm_body_to_inertial
from src.VectorOperations import compute_B_matrix
import matplotlib.pyplot as plt

"""

Resource references:
    https://academicflight.com/articles/equations-of-motion/
    https://aircraftflightmechanics.com/EoMs/Translation.html

"""


class Aircraft():
    """
    States are:
        position: x,y,z (world frame)
        velocity: u,v,w (body frame)
        attitude: phi, theta, psi (world frame)
        angular velocity: p,q,r (body frame)

    Inputs are:
        aileron, elevator, rudder, thrust
    
    """
    def __init__(self, aircraft_params:dict) -> None:
        self.aircraft_params =  aircraft_params
        
        self.velocity_bf = [0, 0, 0] #u,v,w
        self.acc_bf = [0, 0, 0] #u_dot, v_dot, w_dot
        self.angular_velocity_bf = [0, 0, 0] #p,q,r
        self.angular_acc_bf = [0, 0, 0]  #p_dot, q_dot, r_dot or  l,m,n
        
        self.position = [0, 0, 0] #x,y,z
        self.attitudes = [0, 0, 0] #phi, theta, psi
        self.velocity_ef = [0, 0, 0] #x_dot, y_dot, z_dot

        self.position_history = []
        

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
    def compute_moments(self, input_aileron:float,
                        input_elevator:float,
                        input_rudder:float, 
                        force:np.ndarray)-> np.ndarray:
        """
        Returns the L,M,N moments using Rotational Dynamic Equations

        input_aileron in radians
        input_elevator in radians
        input_rudder in radians
        
        Use Austin/Ardupilot
        """
        
        alpha = self.compute_aoa()
        beta = self.compute_beta()
        airspeed = self.aircraft.airspeed
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

        # if self.aircraft.tailsitter or self.aircraft.aerobatic:
        #     effective_airspeed += input_thrust * 20
        #     alpha *= max(0, min(1, 1 - input_thrust))

        rho = 1.225 # kg/m^3
        p = self.aircraft.angular_velocity_bf[0]
        q = self.aircraft.angular_velocity_bf[1]
        r = self.aircraft.angular_velocity_bf[2]

        qbar = 1.0 / 2.0 * rho * pow(effective_airspeed, 2) * s

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
    
    def update_angular_acc(self, moments:np.ndarray) -> None:
        """
        This updates the aircraft p_dot, q_dot, r_dot
        """
        Ixx = self.aircraft.aircraft_params['Ixx']
        Iyy = self.aircraft.aircraft_params['Iyy']
        Izz = self.aircraft.aircraft_params['Izz']

        first_part = (Ixx - Iyy) * self.aircraft.angular_velocity_bf[1] * \
            self.aircraft.angular_velocity_bf[2]
        second_part = (Izz - Iyy) * self.aircraft.angular_velocity_bf[0] * \
            self.aircraft.angular_velocity_bf[2]
        third_part = (Ixx - Izz) * self.aircraft.angular_velocity_bf[1] * \
            self.aircraft.angular_velocity_bf[0]    

        self.aircraft.angular_acc_bf[0] = (moments[0] + first_part) / Ixx
        self.aircraft.angular_acc_bf[1] = (moments[1] + second_part) / Iyy
        self.aircraft.angular_acc_bf[2] = (moments[2] + third_part) / Izz
        
    def update_angular_velocity(self, moments:np.ndarray, 
                                delta_time:float)->None:
        """
        Updates the angular velocity for the body frame : p,q,r
        Refer to this 
        https://academicflight.com/articles/aircraft-attitude-and-euler-angles/

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
        returns the forces in the body frame 
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

    def update_acc(self, forces:np.ndarray) -> None:
        """
        Updates the acceleration of the aircraft in the body frame
        """

        #compute angular rates
        p = self.aircraft.angular_velocity_bf[0]
        q = self.aircraft.angular_velocity_bf[1]
        r = self.aircraft.angular_velocity_bf[2]
        u = self.aircraft.velocity_bf[0]
        v = self.aircraft.velocity_bf[1]
        w = self.aircraft.velocity_bf[2]

        g = 9.80665
        # gravity_earth_frame = np.array([0, 0, g])
        # gravity_body_frame = euler_dcm_inertial_to_body(self.aircraft.attitudes[0],
        #                                                 self.aircraft.attitudes[1],
        #                                                 self.aircraft.attitudes[2]).dot(gravity_earth_frame)
        
        gravity_body_frame = np.array([
            g*np.sin(self.aircraft.attitudes[1]), 
            g*np.sin(self.aircraft.attitudes[0])*np.cos(self.aircraft.attitudes[1]), 
            g*np.cos(self.aircraft.attitudes[0])*np.cos(self.aircraft.attitudes[1])])


        mass = self.aircraft.aircraft_params['mass']
        self.aircraft.acc_bf[0] = (forces[0]/mass) - gravity_body_frame[0] - (q*w)  + (r*v)
        self.aircraft.acc_bf[1] = (forces[1]/mass) + gravity_body_frame[1] - (r*u)  + (p*w)
        self.aircraft.acc_bf[2] = (forces[2]/mass) + gravity_body_frame[2] - (p*v)  + (q*u)


    def update_velocity(self, delta_time:float) -> None:
        """
        Updates the velocity of the aircraft in the body frame
        """
        #make sure self.aircraft.acc_bf is a numpy array
        self.aircraft.acc_bf = np.array(self.aircraft.acc_bf)

        self.aircraft.velocity_bf += self.aircraft.acc_bf*delta_time
        # self.aircraft.velocity_bf[0] = self.aircraft.velocity_bf

        #update airspeed
        self.aircraft.airspeed = np.linalg.norm(self.aircraft.velocity_bf)

    def update_position(self, delta_time:float) -> None:
        """
        update inertial world frame 
        """
        dcm = euler_dcm_body_to_inertial(self.aircraft.attitudes[0],
                                         self.aircraft.attitudes[1],
                                         self.aircraft.attitudes[2])
        self.aircraft.velocity_ef = dcm.dot(self.aircraft.velocity_bf)
        self.aircraft.position += self.aircraft.velocity_ef*delta_time

        self.aircraft.position_history.append(self.aircraft.position)


def get_airplane_params(df:pd.DataFrame) -> dict:
    airplane_params = {}
    for index, row in df.iterrows():
        airplane_params[row["var_name"]] = row["var_val"]

    airplane_params["mass"] = 10 # kg
    
    return airplane_params

if __name__=="__main__":

    df = pd.read_csv("SIM_Plane_h_vals.csv")
    airplane_params = get_airplane_params(df)
    aircraft = Aircraft(airplane_params)

    airspeed = 30 
    aircraft.position = np.array([0, 0, 0], dtype=float)
    aircraft.attitudes = np.array([0, 0, 0], dtype=float)
    aircraft.velocity_bf = np.array([airspeed, 0, 0], dtype=float)
    aircraft.angular_velocity_bf = np.array([0, 0, 0], dtype=float)
    aircraft.angular_acc_bf = np.array([0, 0, 0], dtype=float)
    aircraft.velocity_ef = np.array([airspeed, 0, 0], dtype=float)
    aircraft.airspeed = airspeed

    aircraft_sim = AircraftSim(aircraft)

    #set up the time
    delta_time = 0.01

    #set up the inputs
    input_aileron_rad = np.deg2rad(0)
    input_elevator_rad = np.deg2rad(0)
    input_rudder_rad = 0
    input_thrust = 0 #newtons

    #begin simulation
    n_iter = 1000
    position_history = []
    attitude_history = []
    for i in range(n_iter):
        sim_forces = aircraft_sim.compute_forces(input_aileron_rad,
                                                 input_elevator_rad,
                                                 input_rudder_rad,
                                                 input_thrust)

        sim_moments = aircraft_sim.compute_moments(input_aileron_rad,
                                                    input_elevator_rad,
                                                    input_rudder_rad,
                                                    sim_forces)
        
        aircraft_sim.update_angular_acc(sim_moments)
        aircraft_sim.update_angular_velocity(sim_moments, delta_time)
        aircraft_sim.update_attitude(delta_time)
        aircraft_sim.update_acc(sim_forces)
        aircraft_sim.update_velocity(delta_time)
        aircraft_sim.update_position(delta_time)

        # print("sim_forces: ", sim_forces)
        # print("sim_acc deg/s^2 : ", np.deg2rad(aircraft.acc_bf))
        # print("roll, pitch, yaw: ", np.rad2deg(aircraft.attitudes))
        # print("sim_moments: ", sim_moments)
        # print("earth frame velocity: ", aircraft.velocity_ef)
        # print("body frame velocity: ", aircraft.velocity_bf)
        print("earth frame position: ", aircraft.position)
        current_position = aircraft.position
        print("current position: ", current_position)
        position_history.append([current_position[0],
                                 current_position[1],
                                 current_position[2]])

        attitude_history.append([aircraft.attitudes[0],
                                aircraft.attitudes[1],
                                aircraft.attitudes[2]])

        # print("roll, pitch, yaw: ", np.rad2deg(aircraft.attitudes))

    #convert to numpy array
    x_position = []
    y_position = []
    z_position = []

    for pos in position_history:
        x_position.append(pos[0])
        y_position.append(pos[1])
        z_position.append(-pos[2])
    

    roll = []
    pitch = []
    yaw = []

    for attitude in attitude_history:
        roll.append(np.rad2deg(attitude[0]))
        pitch.append(np.rad2deg(attitude[1]))
        yaw.append(np.rad2deg(attitude[2]))

    #plot 3d 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_position, y_position, z_position, '-o')
    ax.scatter3D(x_position[0], 
                 y_position[1], 
                 -z_position[2],s=15, c='r', label='start')

    ax.legend()
    plt.show()


    fig2 = plt.figure()
    
    # plot as 3 subplots
    ax1 = fig2.add_subplot(311)
    ax1.plot(roll, '-o')
    
    ax2 = fig2.add_subplot(312)

    ax2.plot(pitch, '-o')

    ax3 = fig2.add_subplot(313)
    ax3.plot(yaw, '-o')

    plt.show()



