import pandas as pd
import numpy as np
from src.aircraft.AircraftDynamics import LonAirPlane
from src.Utils import read_lon_matrices, get_airplane_params

df = pd.read_csv("Coeffs.csv")
airplane_params = get_airplane_params(df)
#check if empty params
if not airplane_params:
    raise Exception("airplane params is empty")

lon_airplane = LonAirPlane(airplane_params)

u_0 = 18
theta_0 = 0
A = lon_airplane.compute_A(u_0, theta_0)
B = lon_airplane.compute_B(u_0)

print("at u_0 = {} and theta_0 = {}".format(u_0, theta_0))

#print matrices with 4 decimal places and not in scientific notation
np.set_printoptions(precision=4, suppress=True)
print("A matrix:", A)
print("B matrix:", B)
print("shape of A", A.shape)

#check eigenvalues of A and B  
eigvals_A = np.linalg.eigvals(A)

#visualize the poles of the system
import matplotlib.pyplot as plt
plt.scatter(eigvals_A.real, eigvals_A.imag, marker='x')
plt.show()

print("eigenvalues of A:", eigvals_A)
# print("eigenvalues of B:", eigvals_B)
