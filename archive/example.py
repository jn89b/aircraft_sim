import jsbsim
import time

"""

https://jsbsim-team.github.io/jsbsim/classJSBSim_1_1FGFDMExec.html#a3a5c816fd2db9ddca6c3f209f2b456d0

https://github.com/JSBSim-Team/jsbsim/issues/194

https://jsbsim.sourceforge.net/JSBSim/classJSBSim_1_1FGInitialCondition.html
"""

# Load aircraft configuration
# aircraft = jsbsim.FGFDMExec('./path/to/your/aircraft.xml')
aircraft = jsbsim.FGFDMExec(None)  # Use JSBSim default aircraft data.
aircraft.load_script('scripts/c1723.xml')

# Set initial position (latitude, longitude, altitude)
start_latitude = 0
start_longitude = 0
start_altitude = 0
end_latitude = 0.4
end_longitude = 0.4
end_altitude = 0.4
initial_position = (start_latitude, start_longitude, start_altitude)
aircraft.set_property_value('ic/latitude-deg', initial_position[0])
aircraft.set_property_value('ic/longitude-deg', initial_position[1])
aircraft.set_property_value('ic/altitude-ft', initial_position[2])

# Start simulation
aircraft.run_ic()
aircraft.set_dt(0.1)
# Define end location (latitude, longitude, altitude)
end_position = (end_latitude, end_longitude, end_altitude)

#get all get property values
aircraft.print_property_catalog()

# Simulate flight to the end location
while True:
    # Get current aircraft position
    current_position = (
        aircraft.get_property_value('position/lat-gc-deg'),
        aircraft.get_property_value('position/long-gc-deg'),
        aircraft.get_property_value('position/h-sl-ft')
    )

    # Check if the aircraft has reached the end location
    if current_position[0] >= end_position[0] and current_position[1] >= end_position[1]:
        print("Aircraft reached the end location.")
        break

    # make aircraft go to end location

    # make aicraft go up
    # aircraft.set_property_value('fcs/throttle-cmd-norm', 1)
    # aircraft.set_property_value('fcs/elevator-cmd-norm', 1)

    # Run simulation for one time step
    aircraft.run()

    # Print current position (optional)
    print("Current Position: Latitude {}, Longitude {}, Altitude {} feet".format(
        current_position[0], current_position[1], current_position[2]))

    # Add a delay for visualization purposes
    time.sleep(0.1)

# End simulation
aircraft.close()

