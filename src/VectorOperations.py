import math
import numpy as np
from scipy.spatial.transform import Rotation as R


def body_to_earth_frame(body_euler_angles: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Convert a vector in the body frame to the Earth frame.
    euler_angles: Euler angles describing the rotation from Earth frame to body frame
    vector: Vector in the body frame
    """
    pass


def earth_to_body_frame(earth_euler_angles: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Convert a vector in the Earth frame to the body frame.
    euler_angles: Euler angles describing the rotation from Earth frame to body frame
    vector: Vector in the Earth frame
    """
    pass


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

