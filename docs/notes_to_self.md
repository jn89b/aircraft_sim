# Controlling position
- I can control velocities
- Based on goal position figure out ideal optimal time I should be setting goal velocities 
I have the sign convention flipped for y axis someway somehow, need to fix it

# Guidance 
- Set to LOS angle based on arctangent functions
- As I close in I need to change from LOS angle to the turnpoint heading angle so I can go to the next waypoint
- Need to shift the psi heading one back 
- What is the relationship between interchanging from LOS to turnpoint??

## Low Level Controls stuff
- Get transfer function modeling for the following:
  - Theta/elevator
  - Theta/Thrust
  - Airspeed/elevator
  - Airspeed/Thrust  
  - Phi/aileron 

## 
For radar stuff;
- Write a function with a symbolic value for radar
- Call function out to set the value of the RCS
- 
