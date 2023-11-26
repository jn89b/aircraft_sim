import casadi as ca
import numpy as np

A = ca.SX.sym('A', 6, 6)
states = ca.SX.sym('states', 6, 1)
Ax = ca.mtimes(A, states)
f = ca.Function('dynamics', [A,states], [Ax], 
                ['A', 'states'], ['A_dot'])

a_matrix = np.array([[1, 2, 3, 4, 5, 6],
                     [7, 8, 9, 10, 11, 12],
                     [13, 14, 15, 16, 17, 18],
                     [19, 20, 21, 22, 23, 24],
                     [25, 26, 27, 28, 29, 30],
                     [31, 32, 33, 34, 35, 36]])

states = np.array([0, 1, 2, 3, 4, 5])

print("a_matrix:", a_matrix)

value = f(a_matrix, states)
print("value:", value)

                  