"""
Make sure my orientation matrices are right

- Set BODY roll pitch yaw
- Rotate to INERTIAL frame 

"""

import numpy as np

from scipy.spatial.transform import Rotation as R


phi_body_rad = np.deg2rad(20)
theta_body_rad = np.deg2rad(0)
psi_body_rad = np.deg2rad(0)

original_matrix = np.array([[phi_body_rad, 0, 0],
                            [0, theta_body_rad, 0],
                            [0, 0, psi_body_rad]])

# rotate from body to inertial frame
inertial_rotation = R.from_euler('zyx', [phi_body_rad, theta_body_rad, psi_body_rad], degrees=True)
print("Rotation matrix: ", inertial_rotation.as_matrix())


