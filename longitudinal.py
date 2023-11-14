import numpy as np
from src.Utils import get_airplane_params
import pandas as pd

class Aircraft():
    def __init__(self, aircraft_params) -> None:
        self.aircraft_params = aircraft_params
        self.C_do = self.aircraft_params['c_drag_p']
        self.mass = self.aircraft_params['mass']
        self.C_da = self.aircraft_params['c_lift_a']
        self.C_l0 = self.aircraft_params['c_lift_0']
    
    # def compute_drag(self, velocity: float) -> float:
    #     coefficient = self.aircraft_params
    #     b = coefficient["b"]
    #     s = coefficient["s"]
    #     c_drag_p = coefficient["c_drag_p"]
    #     c_lift_0 = coefficient["c_lift_0"]
    #     c_lift_a0 = coefficient["c_lift_a"]
    #     oswald = coefficient["oswald"]
        
    #     ar = pow(b, 2) / s
    #     c_drag_a = c_drag_p + pow(c_lift_0 + c_lift_a0 * ca_alpha, 2) / (ca.pi * oswald * ar)

    #     return c_drag_a    
    
    def compute_Xu(self, velocity:float) -> None:
        q = 0.5 * 1.225 * pow(velocity, 2)
        return -((0 + (2 * self.C_do)) * q * velocity)/ (self.mass * velocity)
    
    def compute_Xw(self, velocity:float) -> None:
        q = 0.5 * 1.225 * pow(velocity, 2)
        return -((self.C_da - self.C_l0) * q * velocity)/ (self.mass * velocity)
 
    def compute_Xq(self, velocity:float) -> None:

        pass
        
    def longitudinal_state_space(self, velocity) -> None:
        X_u = self.compute_Xu(velocity)
        X_w = self.compute_Xw(velocity)
        self.A = np.array([
            [X_u, X_w, self.X_q, 0.0, self.X_h],
            [self.Z_u, self.Z_w, self.Z_q, self.Z_theta, self.Z_h],
            [self.M_u, self.M_w, self.M_q, self.M_theta, self.M_h],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0]
        ])
    
    


df = pd.read_csv("SIM_Plane_h_vals.csv")
data = get_airplane_params(df)
# Load your CSV data into a NumPy array or pandas dataframe
# Assuming data is stored in a variable called 'data'
# Make sure to replace 'data' with your actual data variable


# Extract stability derivatives
X_u = data['s']
X_w = data['b']
X_q = data['c']
X_theta = data['c_lift_0']
X_h = data['c_lift_a']
X_deltae = data['c_lift_deltae']

Z_u = data['c_y_0']
Z_w = data['c_y_b']
Z_q = data['c_y_p']
Z_theta = data['c_y_deltaa']
Z_h = data['c_y_deltar']
Z_deltae = data['c_drag_deltae']  # Assuming c_drag_deltae corresponds to Z_deltae

M_u = data['c_m_0']
M_w = data['c_m_a']
M_q = data['c_m_q']
M_theta = data['c_m_deltae']
M_h = 0  # Assuming M_h is not provided in your data, set it to 0 or replace it with the correct coefficient
M_deltae = 0  # Assuming M_deltae is not provided in your data, set it to 0 or replace it with the correct coefficient

# Define the state-space matrices
A = np.array([
    [X_u, X_w, X_q, X_theta, X_h],
    [Z_u, Z_w, Z_q, Z_theta, Z_h],
    [M_u, M_w, M_q, M_theta, M_h],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0]
])

B = np.array([
    [X_deltae],
    [Z_deltae],
    [M_deltae],
    [0],
    [0]
])

# Print the resulting A and B matrices
print("A matrix:")
print(A)

print("\nB matrix:")
print(B)
