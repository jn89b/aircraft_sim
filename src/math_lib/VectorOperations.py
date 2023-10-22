import math
import numpy as np
import casadi as ca

"""

https://academicflight.com/articles/kinematics/rotation-formalisms/rotation-matrix/

Library to due all your vector and matrix operations 

"""

def convert_enu_to_ned(enu_vector:np.ndarray) -> np.ndarray:
    """
    This function converts a vector in ENU coordinates to NED coordinates
    """
    ned_vector = np.array([enu_vector[1], enu_vector[0], -enu_vector[2]])
    return ned_vector

def convert_ned_to_enu(ned_vector:np.ndarray) -> np.ndarray:
    """
    This function converts a vector in NED coordinates to ENU coordinates
    """
    enu_vector = np.array([ned_vector[1], ned_vector[0], -ned_vector[2]])
    return enu_vector


def euler_dcm_inertial_to_body(phi_rad:float, 
                               theta_rad:float, 
                               psi_rad:float) -> np.ndarray:
    """
    This computes the DCM matrix going from inertial to body frame
    """
    # Compute the direction cosine matrix elements
    cos_phi = np.cos(phi_rad)
    sin_phi = np.sin(phi_rad)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    cos_psi = np.cos(psi_rad)
    sin_psi = np.sin(psi_rad)
    
    # Compute the DCM elements    
    dcm = np.array([[cos_theta * cos_psi, cos_theta * sin_psi, -sin_theta],
                    [sin_phi * sin_theta * cos_psi - cos_phi * sin_psi, 
                     sin_phi * sin_theta * sin_psi + cos_phi * cos_psi, 
                     sin_phi * cos_theta],
                    [cos_phi * sin_theta * cos_psi + sin_phi * sin_psi, 
                     cos_phi * sin_theta * sin_psi - sin_phi * cos_psi, 
                    cos_phi * cos_theta]])
        
    return dcm

def euler_dcm_body_to_inertial(phi_rad:float,
                               theta_rad:float,
                               psi_rad:float) -> np.ndarray:

    # Compute the direction cosine matrix elements
    dcm_inert_to_body = euler_dcm_inertial_to_body(phi_rad, theta_rad, psi_rad)
    #return the inverse of this 
    return dcm_inert_to_body.T

def compute_B_matrix(phi_rad:float, theta_rad:float, psi_rad:float) -> np.ndarray:
    """
    Computes the B matrix for the body frame
    """
    # Compute the direction cosine matrix elements
    cos_phi = np.cos(phi_rad)
    sin_phi = np.sin(phi_rad)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    # Compute the B matrix elements
    B = np.array([[cos_theta, sin_phi*sin_theta, cos_phi*sin_theta],
                  [0, cos_phi*cos_theta, -sin_phi*cos_theta],
                  [0, sin_phi, cos_phi]])
    B = (1/cos_theta) * B

    return B    


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

class Vector3D():
    def __init__(self, x:float, y:float, z:float) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.array = np.array([self.x, self.y, self.z])

    def update_array(self) -> None:
        self.array = np.array([self.x, self.y, self.z])

    def update_positions(self) -> None:
        self.x = self.array[0]
        self.y = self.array[1]
        self.z = self.array[2]
