import jsbsim

print(jsbsim.get_default_root_dir())
fdm = jsbsim.FGFDMExec(None)  # Use JSBSim default aircraft data.
fdm.load_script('scripts/c1723.xml')
fdm.run_ic()

# Run the simulation for 10 seconds.
fdm.set_sim_time(10.0)

# set a start condition
fdm.set_property_value('ic/h-sl-ft', 10000)
fdm.set_property_value('ic/terrain-elevation-ft', 0)
fdm.set_property_value('ic/latitude-geod-deg', 0)
fdm.set_property_value('ic/longitude-gc-deg', 0)


while fdm.run():
  # Do something with the state of the simulation.
    print(fdm.get_property_value('velocities/v-down-fps'))
    