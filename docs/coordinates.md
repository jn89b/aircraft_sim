# How do the coordinates work in Simulation

- If utilizing terrain with the global planner the terrain must be instianted first
- From there the global planner will take in the terrain information
- Planning from the global planner will be planned in its own reference frame of 0,0 to the maximum bounds of the terrain map
- Once the global planner returns the path, a conversion must be made to switch from the global planner coordinate frame to the terrain coordinate frame, this can be done by utilizing the DataParser class method "format_traj_data_with_terrain"


## Procedure to set up environment

- Load the terrain
- Load the obstacles
- Load radars
- Create graph and:
  - Set terrain
  - Set obstacles
  - Set radars
  - raytrace with radars
- Call out SAS
  - set to graph and plan
    - when expanding moves:
      - check if move is in bounds
      - check if move not in terrain
      - check if move not in obstacles
      - check if move in fov of radar:
        - if in compute rcs 
      - check constraints of path based on max flight endurance 
