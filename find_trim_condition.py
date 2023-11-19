import numpy as np
import pandas as pd
import casadi as ca
import matplotlib.pyplot as plt

from src.aircraft.AircraftDynamics import AircraftCasadi
from src.Utils import get_airplane_params


"""
Steady State Straight and Level Trimmed Flight

Utilizing optimal control to find the trim condition for 
the aircraft to fly straight and level

# https://www.youtube.com/watch?v=YzZI1V2mJw8&ab_channel=ChristopherLum

"""

if __name__=="__main__":

    df = pd.read_csv("SIM_Plane_h_vals.csv")
    airplane_params = get_airplane_params(df)
    print("mass", airplane_params['mass'])
    aircraft = AircraftCasadi(airplane_params)

    f = aircraft.set_state_space()

    init_x = 0
    init_y = 0
    init_z = 0
    init_u = 18
    init_v = 0
    init_w = 0
    init_phi = 0
    init_theta = 0
    init_psi = 0
    init_p = 0
    init_q = 0
    init_r = 0

    init_el = np.deg2rad(0)
    init_al = np.deg2rad(0)
    init_rud = 0
    init_throttle = 0

    goal_x = 0
    goal_y = 0
    goal_z = 0
    goal_phi = np.deg2rad(0)
    goal_theta = np.deg2rad(0)
    goal_psi = np.deg2rad(0)

    desired_airspeed = 25


    states = np.array([
        init_x, init_y, init_z,
        init_u, init_v, init_w,
        init_phi, init_theta, init_psi,
        init_p, init_q, init_r
    ])
    controls = np.array([init_al, init_el, init_rud, init_throttle])

    #set max attitude
    max_phi = np.deg2rad(45)
    min_phi = np.deg2rad(-45)
    max_theta = np.deg2rad(25)
    min_theta = np.deg2rad(-25)
    min_vel = 15
    max_vel = 30

    #set the control surface limits
    max_control_surface = np.deg2rad(25)
    min_control_surface = np.deg2rad(-25)
    min_throttle = 20 # Newtons
    max_throttle = 200 # Newtons

    # Optimal control problem
    opti = ca.Opti()
    dt = 1/100 # Time step
    N = 50
    t_init = 0 # Initial time

    # Define the states over the optimization problem
    X = opti.variable(aircraft.num_states, N+1) # State vector
    U = opti.variable(aircraft.num_controls, N) # Control vector
    
    x0 = opti.parameter(aircraft.num_states, 1) # Initial state
    xF = opti.parameter(aircraft.num_states, 1) # Final state
    
    u0 = opti.parameter(aircraft.num_controls, 1) # Initial control input

    # set initial value
    opti.set_value(x0, np.array([
                                 init_x, init_y, init_z, 
                                 init_u, init_v, init_w, 
                                 init_phi, init_theta, init_psi, 
                                 init_p, init_q, init_r]))
    
    opti.set_value(xF, np.array([
                                 0, 0, 0, 
                                 25, 0, 0, 
                                 0, 0, 0, 
                                 0, 0, 0]))
    opti.set_value(u0, controls)

    # set initial and terminal constraints
    opti.subject_to(X[:,0] == x0)
    # opti.subject_to(X[:,-1] == xF)
    opti.subject_to(U[:,0] == u0)
    
    #angular rates should be 0 for trim
    opti.subject_to(X[9,:] == 0)
    opti.subject_to(X[10,:] == 0)
    opti.subject_to(X[11,:] == 0)
    
    # set cost function 
    # minimize the error between the final state and the goal state
    #set weights for the cost function as a array
    weights = np.array([1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0])
    
    weights_controls = np.array([1.0, 1.0, 1.0, 1.0])

    #set the cost function to minimize error and the control inputs
    # state_error = weights*ca.sumsqr((X[:, -1] - xF))
    #cost_fn = ca.sumsqr(weights * (X[:, -1] - xF) + ca.sumsqr(weights_controls*U**2))
    # magnitude airspeed error
    
    cost = 0
    
    #compute airstream velocity
    v_a = ca.sqrt(X[3,:]**2 + X[4,:]**2 + X[5,:]**2)
    
    alpha = ca.atan(X[5,:]/X[3,:])
    theta = X[7,:]
    gamma = theta - alpha
    v = X[4,:]

    #minimze teh change of height
    cost_fn = 0
    cost_fn = cost_fn + ca.sumsqr(X[3,:] - desired_airspeed)  + ca.sumsqr(v)
    
    opti.minimize(cost_fn) 

    for k in range(N):
        
        k1 = f(X[:,k], U[:,k])
        k2 = f(X[:,k] + dt/2 * k1, U[:,k])
        k3 = f(X[:,k] + dt/2 * k2, U[:,k])
        k4 = f(X[:,k] + dt * k3, U[:,k])
        x_next = X[:,k] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        opti.subject_to(X[:,k+1]==x_next) # Use initial state
        
        dz = X[2,k+1] - X[2,k]
    
        #minimize gamma, theta, velocity and error of desired velocity
        cost_fn += ca.sumsqr(gamma) + ca.sumsqr(theta) + ca.sumsqr(v) + ca.sumsqr(v_a - desired_airspeed) + ca.sumsqr(dz)

    # solve the optimization problem
    opts = {
        'ipopt': {
            'max_iter': 1000,
            'print_level': 2,
            'acceptable_tol': 1e-2,
            'acceptable_obj_change_tol': 1e-2,
            'hessian_approximation': 'limited-memory',  # Set Hessian approximation method here
        },
        'print_time': 1
    }

    opti.solver('ipopt', opts)#, {'ipopt': {'print_level': 0}})

    #Initial Trajectory    
    sol = opti.solve()    
    print("Opti solver", opti)
    print(sol.value(X))
    #debugging
    opti.debug.value(X)

    #plot everything
    
    # plot position in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(sol.value(X[0,:]), sol.value(X[1,:]), -sol.value(X[2,:]) , '-o')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')

    #plot starting and ending points
    ax.scatter(sol.value(X[0,0]), sol.value(X[1,0]), 
               -sol.value(X[2,0]), c='r', marker='o', label='start')
    # ax.scatter(sol.value(X[0,-1]), sol.value(X[1,-1]), 
    #            -sol.value(X[2,-1]), c='c', marker='o', label='end')

    #plot the goal
    print("goal", goal_x, goal_y, goal_z)
    ax.scatter(goal_x, goal_y, -goal_z, c='g', marker='x', label='goal')

    #set axis equal
    max_range = np.array([sol.value(X[0,:]).max()-sol.value(X[0,:]).min(), sol.value(X[1,:]).max()-sol.value(X[1,:]).min(), sol.value(X[2,:]).max()-sol.value(X[2,:]).min()]).max() / 2.0
    mean_x = sol.value(X[0,:]).mean()
    mean_y = sol.value(X[1,:]).mean()
    mean_z = sol.value(X[2,:]).mean()
    max_z_range = np.array([sol.value(X[2,:]).max()-sol.value(X[2,:]).min()]).max() / 2.0
    # ax.set_xlim(mean_x - max_range, mean_x + max_range)
    # ax.set_ylim(mean_y - max_range, mean_y + max_range)
    # ax.set_zlim(mean_z - max_z_range, mean_z + 30)
    ax.legend()

    t_vec = np.linspace(t_init, t_init + N*dt, N)
    #plot control inputs in subplots
    fig, axs = plt.subplots(4,1)
    axs[0].plot(t_vec, np.rad2deg(sol.value(U[0,:])))
    axs[0].set_ylabel('Aileron [deg]')
    axs[1].plot(t_vec, np.rad2deg(sol.value(U[1,:])))
    axs[1].set_ylabel('Elevator [deg]')
    axs[2].plot(t_vec, np.rad2deg(sol.value(U[2,:])))
    axs[2].set_ylabel('Rudder [deg]')
    axs[3].plot(t_vec,sol.value(U[3,:]))
    axs[3].set_ylabel('Throttle [N]')

    # plot the attitudes
    t_vec = np.linspace(t_init, t_init + N*dt, N+1)
    fig, axs = plt.subplots(3,1)
    axs[0].plot(t_vec, np.rad2deg(sol.value(X[6,:])))
    axs[0].set_ylabel('Roll [deg]')
    axs[1].plot(t_vec, np.rad2deg(sol.value(X[7,:])))
    axs[1].set_ylabel('Pitch [deg]')
    axs[2].plot(t_vec, np.rad2deg(sol.value(X[8,:])))
    axs[2].set_ylabel('Yaw [deg]')
    axs[2].set_xlabel('Time [s]')

    #plot the velocities
    fig, axs = plt.subplots(3,1)
    axs[0].plot(t_vec, sol.value(X[3,:]))
    axs[0].set_ylabel('u [m/s]')
    axs[1].plot(t_vec, sol.value(X[4,:]))
    axs[1].set_ylabel('v [m/s]')
    axs[2].plot(t_vec, sol.value(X[5,:]))
    axs[2].set_ylabel('w [m/s]')
    axs[2].set_xlabel('Time [s]')
    
    #plot control inputs in subplots
    fig, axs = plt.subplots(3,1)
    axs[0].plot(t_vec, np.rad2deg(sol.value(X[9,:])))
    axs[0].set_ylabel('p [deg/s]')
    axs[1].plot(t_vec, np.rad2deg(sol.value(X[10,:])))
    axs[1].set_ylabel('q [deg/s]')
    axs[2].plot(t_vec, np.rad2deg(sol.value(X[11,:])))
    axs[2].set_ylabel('r [deg/s]')
    
    #plot airspeed
    fig, axs = plt.subplots(1,1)
    airspeed = np.sqrt(sol.value(X[3,:])**2 + sol.value(X[4,:])**2 + sol.value(X[5,:])**2)
    axs.plot(t_vec, airspeed)

    #print the cost function
    print("Cost Function", sol.value(cost_fn))

    #print final states and controls
    print("Final States")
    print("x", sol.value(X[0,-1]))
    print("y", sol.value(X[1,-1]))
    print("z", sol.value(X[2,-1]))
    print("u", sol.value(X[3,-1]))
    print("v", sol.value(X[4,-1]))
    print("w", sol.value(X[5,-1]))
    print("phi", np.rad2deg(sol.value(X[6,-1])))
    print("theta", np.rad2deg(sol.value(X[7,-1])))
    print("psi", np.rad2deg(sol.value(X[8,-1])))
    print("p", np.rad2deg(sol.value(X[9,-1])))
    print("q", np.rad2deg(sol.value(X[10,-1])))
    print("r", np.rad2deg(sol.value(X[11,-1])))
    print("Final Controls")
    print("aileron", np.rad2deg(sol.value(U[0,-1])))
    print("elevator", np.rad2deg(sol.value(U[1,-1])))
    print("rudder", np.rad2deg(sol.value(U[2,-1])))
    print("throttle", sol.value(U[3,-1]))
    
    #save the final values  and controls to a csv file
    #create a dictionary
    final_states = {"x": sol.value(X[0,-1]),
                    "y": sol.value(X[1,-1]),
                    "z": sol.value(X[2,-1]),
                    "u": sol.value(X[3,-1]),
                    "v": sol.value(X[4,-1]),
                    "w": sol.value(X[5,-1]),
                    "phi": np.rad2deg(sol.value(X[6,-1])),
                    "theta": np.rad2deg(sol.value(X[7,-1])),
                    "psi": np.rad2deg(sol.value(X[8,-1])),
                    "p": np.rad2deg(sol.value(X[9,-1])),
                    "q": np.rad2deg(sol.value(X[10,-1])),
                    "r": np.rad2deg(sol.value(X[11,-1]))}
    
    final_controls = {"delta_a": np.rad2deg(sol.value(U[0,-1])),
                        "delta_e": np.rad2deg(sol.value(U[1,-1])),
                        "delta_r": np.rad2deg(sol.value(U[2,-1])),
                        "delta_t": sol.value(U[3,-1])}
    
    
    #save the dictionary to a csv file
    #convert the dictionary to a dataframe
    df_states = pd.DataFrame(final_states, index=[0])
    df_controls = pd.DataFrame(final_controls, index=[0])
    
    #save the dataframe to a csv file
    df_states.to_csv("final_states.csv")
    df_controls.to_csv("final_controls.csv")
    
    plt.show()


