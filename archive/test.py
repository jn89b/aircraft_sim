import numpy as np

def runge_kutta(f, x0, y0, h, n):
    """
    Runge-Kutta method for solving ordinary differential equations.
    
    Parameters:
        f: function representing the derivative dy/dx
        x0, y0: initial conditions
        h: step size
        n: number of steps
        
    Returns:
        Arrays containing x and y values
    """
    x_values = [x0]
    y_values = [y0]
    
    for i in range(n):
        k1 = h * f(x0, y0)
        k2 = h * f(x0 + h/2, y0 + k1/2)
        k3 = h * f(x0 + h/2, y0 + k2/2)
        k4 = h * f(x0 + h, y0 + k3)
        
        y0 = y0 + (k1 + 2*k2 + 2*k3 + k4) / 6
        x0 = x0 + h
        
        x_values.append(x0)
        y_values.append(y0)
    
    return np.array(x_values), np.array(y_values)

# Function representing the derivative dy/dx for the cannonball example
def cannonball_derivative(x, y):
    g = 9.81  # gravitational acceleration (m/s^2)
    air_density = 1.225  # air density at sea level (kg/m^3)
    cross_sectional_area = np.pi * (0.1**2)  # cross-sectional area of the cannonball (m^2)
    drag_coefficient = 0.47  # drag coefficient of the cannonball
    mass = 4.5  # mass of the cannonball (kg)
    
    # Equation for air resistance (drag) force: F_drag = 0.5 * Cd * A * rho * v^2
    drag_force = 0.5 * drag_coefficient * cross_sectional_area * air_density * (y**2)
    
    # Equation for gravitational force: F_gravity = m * g
    gravity_force = mass * g
    
    # Equation for the derivative dy/dx (velocity) considering drag and gravity
    dy_dx = -(drag_force - gravity_force) / mass
    
    return dy_dx

# Initial conditions
x0 = 0  # initial time
y0 = 0  # initial velocity

# Step size and number of steps
h = 0.01  # step size
n = 1000  # number of steps

# Runge-Kutta method
x_values, y_values = runge_kutta(cannonball_derivative, x0, y0, h, n)

# Print the results
for i in range(len(x_values)):
    print(f"Time: {x_values[i]:.2f} seconds, Velocity: {y_values[i]:.2f} m/s")
