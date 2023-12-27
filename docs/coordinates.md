#  How do the Coordinates work in Simulation
- Terrain specified with bounds from min lat, min lon to max lat and max lon
- From there derive cartesian coordinates 
- Global planner will utilize offset 

When SAS is utilized, any position from its map will be queried to 

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
