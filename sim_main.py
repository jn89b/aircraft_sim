import pandas as pd

from src.AircraftDynamics import AircraftDynamics
from src.Aircraft import AircraftInfo
from src.utils import get_airplane_params


if __name__=="__main__":
    df = pd.read_csv("SIM_Plane_h_vals.csv")
    airplane_params = get_airplane_params(df)
    
    init_states = {'x':0.0, 'y':0.0, 'z':0.0,
                   'u':0.0, 'v':0.0, 'w':0.0,
                   'phi':0.0, 'theta':0.0, 'psi':0.0,
                   'p':0.0, 'q':0.0, 'r':0.0}
    
    controls = {'delta_e':0.0, 
                'delta_a':0.0, 
                'delta_r':0.0, 
                'delta_t':0.0}
    
    aircraft_info_euler = AircraftInfo(
        airplane_params,
        init_states,
        controls)
    
    aircraft_info_rk = AircraftInfo(
        airplane_params,
        init_states,
        controls)
    

    aircraft_dynamics_eulers = AircraftDynamics(aircraft_info_euler)
    aircraft_dynamics_rk = AircraftDynamics(aircraft_info_rk)

    print(aircraft_info_euler.states)

    N = 50
    dt = 0.1

    input_aileron = controls['delta_a']
    input_elevator = controls['delta_e']
    input_rudder = controls['delta_r']
    input_throttle = controls['delta_t']

    euler_states = []
    rk_states = []

    for i in range(N):

        new_states_eulers = aircraft_dynamics_eulers.eulers(
            input_aileron,
            input_elevator,
            input_rudder,
            input_throttle,
            aircraft_info_euler.states,
            dt
        )

        new_states_rk = aircraft_dynamics_rk.rk45(
            input_aileron,
            input_elevator,
            input_rudder,
            input_throttle,
            aircraft_info_rk.states,
            dt
        )
        aircraft_info_rk.update_states(new_states_rk)
        aircraft_info_euler.update_states(new_states_eulers)

        euler_states.append(new_states_eulers)
        rk_states.append(new_states_rk)

    euler_states = pd.DataFrame(euler_states)
    rk_states = pd.DataFrame(rk_states)

    #set column names
    euler_states.columns = ['x','y','z','u','v','w','phi','theta','psi','p','q','r']
    rk_states.columns = ['x','y','z','u','v','w','phi','theta','psi','p','q','r']

    print(euler_states)

    # # plot the results
    import matplotlib.pyplot as plt
    plt.figure()
    
    #3d plot 
    from mpl_toolkits.mplot3d import Axes3D
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':'3d'})
    ax.plot(euler_states['x'], euler_states['y'], -euler_states['z'], 'o-', 
            label='Euler',)
    ax.plot(rk_states['x'], rk_states['y'], -rk_states['z'], 'x-' ,
            label='RK45')
    ax.legend()

    plt.show() 
