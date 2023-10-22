"""
Simple sim where I define all my coefficients for aircraft

Make the aircraft move based on input command of aileron, elevator, rudder and throttle

https://www.youtube.com/watch?v=jSf_nMGg_dI&list=PLIwDIOqR-ET0kKguPqG-2CBQyZdG7Ck9r&index=23&ab_channel=AeroAcademy
"""

import numpy as np
import matplotlib.pyplot as plt
import math 
import pandas as pd

from src.VectorOperations import Vector3D, DirectionCosineMatrix

AIR_DENSITY = 1.225 # kg/m^3
GRAVITY = 9.81 # m/s^2

# class Vector3D():
#     def __init__(self, x:float, y:float, z:float) -> None:
#         self.x = x
#         self.y = y
#         self.z = z
#         self.array = np.array([self.x, self.y, self.z])


class Aircraft():
    def __init__(self, aircraft_params:dict) -> None:
        self.aircraft_params = aircraft_params

        self.tailsitter = False
        self.aerobatic = False
        self.dcm = DirectionCosineMatrix()
        
        # set diagonals of dcm


        self.gyro = Vector3D(0, 0, 0) # rad/s
        self.airspeed = 20 # m/s

        self.accel_body = Vector3D(0, 0, 0) # m/s^2 acceleration of aircraft in body frame
        self.velocity_bf = Vector3D(self.airspeed, 0, 0) # velocity of aircraft in body frame
        self.velocity_air_bf = Vector3D(self.airspeed, 0, 0) # velocity of aircraft in body frame

        self.velocity_ef = Vector3D(self.airspeed, 0, 0) # velocity of aircraft in earth frame
        self.position_ef = Vector3D(0, 0, 0) # position of aircraft in earth frame
        self.velocity_air_ef = Vector3D(self.airspeed, 0, 0) # velocity of air in earth frame

    def get_attitudes(self):
        #return diagonals of dcm
        return self.dcm.matrix[0, 0], self.dcm.matrix[1, 1], self.dcm.matrix[2, 2]
    

class AircraftDynamics():

    def __init__(self, aircraft:Aircraft) -> None:
        self.aircraft = aircraft
        self.angle_of_attack_rad = None
        self.beta_rad = None

    def compute_aoa(self):
        """computes the angle of attack in radians"""
        alpha = np.arctan2(self.aircraft.velocity_bf.z, self.aircraft.velocity_bf.x)

        return np.arctan2(self.aircraft.velocity_bf.z, self.aircraft.velocity_bf.x)

    def compute_beta(self):
        """computes the sideslip angle in radians"""
        return np.arctan2(self.aircraft.velocity_bf.y, self.aircraft.velocity_bf.x)

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
        """computes the drag coefficient with a linear and flat plate model"""
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
    
    def get_forces(self, aileron:float, elevator:float, rudder:float) -> Vector3D:
        alpha_rad = self.angle_of_attack_rad
        beta_rad = self.beta_rad
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
        rho = AIR_DENSITY

        # get lift and drag alpha coefficients 
        c_lift_a = self.lift_coeff(alpha_rad)
        c_drag_a = self.drag_coeff(alpha_rad)

        # coefficients to the body frame
        c_x_a = -c_drag_a*np.cos(alpha_rad) + c_lift_a*np.sin(alpha_rad)
        c_x_q = -c_drag_q*np.cos(alpha_rad) + c_lift_q*np.sin(alpha_rad)
        c_z_a = -c_drag_a*np.sin(alpha_rad) - c_lift_a*np.cos(alpha_rad)
        c_z_q = -c_drag_q*np.sin(alpha_rad) - c_lift_q*np.cos(alpha_rad)

        #compute angular rates
        p = self.aircraft.gyro.x
        q = self.aircraft.gyro.y
        r = self.aircraft.gyro.z

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
                       c_drag_deltae*np.cos(alpha_rad)*abs(elevator) + \
                        c_lift_deltae*np.sin(alpha_rad)*elevator)
            ay = qbar*(c_y_0 + c_y_b*beta_rad + c_y_p*b*p/(2*airspeed) + \
                       c_y_r*b*r/(2*airspeed) + c_y_deltaa*aileron + c_y_deltar*rudder)
            az = qbar*(c_z_a + c_z_q*c*q/(2*airspeed) - \
                 c_drag_deltae*np.sin(alpha_rad)*abs(elevator) - \
                    c_lift_deltae*np.cos(alpha_rad)*elevator)
            
        return Vector3D(ax, ay, az)


    def get_moment(self, input_aileron, 
                   input_elevator, input_rudder, 
                   input_thrust, force) -> Vector3D:
        alpha = self.angle_of_attack_rad
        beta = self.beta_rad
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

        CGOffset = Vector3D(CGOffset_x, CGOffset_y, CGOffset_z)

        if self.aircraft.tailsitter or self.aircraft.aerobatic:
            effective_airspeed += input_thrust * 20
            alpha *= max(0, min(1, 1 - input_thrust))

        rho = AIR_DENSITY # getting from a gyro sensor
        p = self.aircraft.gyro.x
        q = self.aircraft.gyro.y
        r = self.aircraft.gyro.z

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

        la += CGOffset.y * force.z - CGOffset.z * force.y
        ma += -CGOffset.x * force.z + CGOffset.z * force.x
        na += -CGOffset.y * force.x + CGOffset.x * force.y

        return Vector3D(la, ma, na)

    def compute_forces(self, inputs:dict) -> Vector3D:
        """
        computes the forces acting on the aircraft
        Throttle is set as a value of 0 to 1 based on max thrust force
        
        Aileron = +/- 25 deg
        Elevator = +/- 25 deg
        Rudder = +/- 25 deg

        Thrust = 0 to 1  -> Voltage to motor 
        
        """
        # need to figure out throttle, aileron, elevator, rudder
        thrust = inputs["input_thrust_n"]
        aileron = inputs["input_aileron_rad"]
        elevator = inputs["input_elevator_rad"]
        rudder = inputs["input_rudder_rad"]

        self.angle_of_attack_rad = self.compute_aoa()
        self.beta_rad = self.compute_beta()

        force = self.get_forces(aileron, elevator, rudder)
        rot_accel = self.get_moment(aileron, elevator, rudder, thrust, 
                                    force)

        self.aircraft.accel_body.x = (force.x + thrust) / self.aircraft.aircraft_params["mass"]
        self.aircraft.accel_body.y = force.y / self.aircraft.aircraft_params["mass"]
        self.aircraft.accel_body.z = force.z  / self.aircraft.aircraft_params["mass"]
        self.aircraft.accel_body.update_array()

        return rot_accel

    def update_dynamics(self, rot_accel:Vector3D, dt:float):
        """
        Updates the simulation attitude and relative position
        """
        delta_time = dt

        # update rotational rates in body frame
        self.aircraft.gyro.array += rot_accel.array * delta_time
        self.aircraft.gyro.update_positions()
        self.aircraft.gyro.x = max(
            min(self.aircraft.gyro.x, math.radians(2000.0)), -math.radians(2000.0))
        self.aircraft.gyro.y = max(
            min(self.aircraft.gyro.y, math.radians(2000.0)), -math.radians(2000.0))
        self.aircraft.gyro.z = max(
            min(self.aircraft.gyro.z, math.radians(2000.0)), -math.radians(2000.0))
        self.aircraft.gyro.update_array()

        # update attitude -> this is roll, pitch, yaw
        self.aircraft.dcm.rotate(self.aircraft.gyro.array, delta_time)
        self.aircraft.dcm.normalize()

        accel_earth = self.aircraft.dcm.matrix.dot(self.aircraft.accel_body.array)
        accel_earth += [0, 0, GRAVITY]

        # work out acceleration as seen by the accelerometers. It sees the kinematic
        # acceleration (ie. real movement), plus gravity
        
        #accel_body = dcm.transposed() * (accel_earth + Vector3f(0.0f, 0.0f, -GRAVITY_MSS));
        accel_earth_with_gravity = accel_earth + [0, 0, -GRAVITY]
        self.aircraft.accel_body.array = self.aircraft.dcm.transpose().dot(
            accel_earth_with_gravity)
        
        self.aircraft.velocity_ef.array += accel_earth * delta_time
        self.aircraft.velocity_ef.update_positions()

        self.aircraft.position_ef.array += self.aircraft.velocity_ef.array * delta_time
        self.aircraft.position_ef.update_positions()

        # update airspeed
        self.aircraft.airspeed = np.linalg.norm(self.aircraft.velocity_ef.array)


    def update(self, rot_accel:Vector3D, dt:float, inputs:dict):
        """
        Integrates the aircraft dynamics
        Inputs: aileron, elevator, rudder, throttle

        thrust = inputs["input_thrust_n"]
        aileron = inputs["input_aileron_rad"]
        elevator = inputs["input_elevator_rad"]
        rudder = inputs["input_rudder_rad"]

        """
        rotation_accel = self.compute_forces(inputs)
        self.update_dynamics(rotation_accel, dt)

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

    aircraft_dynamics = AircraftDynamics(aircraft)
    
    #controller will take care of this
    inputs = {"input_thrust_n": 20.0, 
              "input_aileron_rad": np.deg2rad(0), 
              "input_elevator_rad": np.deg2rad(0), 
              "input_rudder_rad": 0}
    dt = 0.1
    t_sim = 5
    n_iter = int(t_sim/dt)
    t_curr = 0
  
    dcm_matrices = []
    for i in range(n_iter):
        aircraft_dynamics.update(Vector3D(0, 0, 0), dt, inputs)
        t_curr += dt
        #print("t_curr", t_curr)
        attitudes = aircraft.get_attitudes()
        dcm_matrices.append(aircraft.dcm.matrix) 
        print("attitudes", np.rad2deg(attitudes))
    
    #plot and animate the Direction Cosine Matrices from the simulation
    dcm_matrices = np.array(dcm_matrices)

