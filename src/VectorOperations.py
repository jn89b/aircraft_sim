import math
import numpy as np
from scipy.spatial.transform import Rotation as R

"""
Make sure my orientation matrices are right

- Set BODY roll pitch yaw
- Rotate to INERTIAL frame 

https://academicflight.com/articles/kinematics/rotation-formalisms/rotation-matrix/

"""


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

class DirectionCosineMatrix:
    def __init__(self):
        self.matrix = np.identity(3)  # Initialize DCM as an identity matrix

    def rotate(self, gyro, delta_time):
        """
        Rotate the DCM based on gyroscopic measurements.
        gyro: 3D angular velocity vector (gyro.x, gyro.y, gyro.z)
        delta_time: Time interval for the rotation
        https://academicflight.com/articles/kinematics/rotation-formalis ms/rotation-matrix/
        """

        gyro_matrix = np.array([[0, -gyro[2], gyro[1]],
                                [gyro[2], 0, -gyro[0]],
                                [-gyro[1], gyro[0], 0]])
                
        # self.matrix += gyro_matrix.dot(self.matrix) * delta_time
        


    def normalize(self):
        """
        Normalize the DCM to maintain its orthogonality and determinant
        """
        norm = np.linalg.norm(self.matrix)
        self.matrix /= norm

    def transpose(self):
        """
        Transpose the DCM.
        """
        return np.transpose(self.matrix)



