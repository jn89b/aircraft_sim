"""
Make sure my orientation matrices are right

- Set BODY roll pitch yaw
- Rotate to INERTIAL frame 

https://academicflight.com/articles/kinematics/rotation-formalisms/rotation-matrix/

"""

import numpy as np


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
    dcm = np.array([[cos_theta * cos_psi, 
                     sin_phi * sin_theta * cos_psi - cos_phi * sin_psi, 
                     cos_phi * sin_theta * cos_psi + sin_phi * sin_psi
                     ],
                    [cos_theta * sin_psi, 
                     sin_phi * sin_theta * sin_psi + cos_phi * cos_psi, 
                     cos_phi * sin_theta * sin_psi - sin_phi * cos_psi
                     ],
                    [-sin_theta, 
                     sin_phi * cos_theta, 
                     cos_phi * cos_theta
                     ]])
    
    return dcm

def euler_dcm_body_to_inertial(phi_rad:float,
                               theta_rad:float,
                               psi_rad:float) -> np.ndarray:
    
    # Compute the direction cosine matrix elements
    dcm_inert_to_body = euler_dcm_inertial_to_body(phi_rad, theta_rad, psi_rad)
    #return the inverse of this 
    return dcm_inert_to_body.T


# Example usage:
phi = 45  # Roll angle in degrees
theta = 30  # Pitch angle in degrees
psi = 60  # Yaw angle in degrees

dcm_matrix = euler_dcm_inertial_to_body(phi, theta, psi)
print("DCM Matrix:")
print(dcm_matrix)

another_dcm_matrix = euler_dcm_body_to_inertial(phi, theta, psi)

#DCM from inertial to body

