# Useful links 
https://www.youtube.com/watch?v=SOM98EpErT8&list=PLIwDIOqR-ET0kKguPqG-2CBQyZdG7Ck9r&index=2&ab_channel=AeroAcademy
- Developing a flight simulation  

# Matrix Operations
- Be able to compute body frame to world frame 
- Be able o compute world frame to body frame
- Translate a position and rotate it 
  - [x] Verify that it works visually
  - [x] Set 


# Aircraft Simulator

- Calculate Aerodynamic Forces

- Predict velocity and angular velocity 

- Predict Torque and Force prediction 

- Predice force and toqrue prediction 


# Guidance Code
- Test with opti stack framework first then add complexity 
- Compute waypoints 
- From waypoints 
  - compute attitude desired 
  - set position desired 

MPC FORMULATION
- Set states as :
  - x y z
  - u v w # body frame rate
  - phi theta psi # 
  - p q r #body frame rates 

- Rotate and integrate 

- Set controls as:
  - Throttle
  - daileron 
  - delevator
  - drudder

- Set constraints for controls:
  - Max angle flaps

- Set constraints for states:
  - u v w
  - phi theta psi
  - p q r



# Modeling Bullshit
- Optimize to find trim conditions for aircraft
- With this linearize aircraft 
- Go to nonlinear equations to linear
- Get the matrix bullshit then save as coefficients 
  