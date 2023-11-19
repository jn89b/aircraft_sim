- Missing the following in the first sheet:
  - c_drag_0
  - lift coefficient calculation,C_l:
    - Changes as a function of alpha but for now steady level flight
      - Using equation $L=W=mg$
    - c_drag_a computations from spreadsheet:
      - Where does the 6 come from aspect ratio? 
      - Forgot a parenthesis
  - M_de calcs
    - Used airspeed should be dynamic
    - Should be -cm_deltae 

- I don't think dynamic A's and B's will work:
  - Would need to trim then find the A's and B's for that then adjust 
- Have better results setting using Nominal A and B matrices 


- Aircraft naturally likes to climb at trim condition  
- How do I regulate that? 

# Regulating Airspeed and altitude 
https://eng.libretexts.org/Bookshelves/Aerospace_Engineering/Aerodynamics_and_Aircraft_Performance_(Marchman)/05%3A_Altitude_Change-_Climb_and_Guide
- I need to set a correct theta based on trim condition and the airspeed of the aircraft 
- Good goal pitch is -7.2 degrees for system to control the airspeed and the altitude
- What is this relationship??? 
  - Related towards the pitching moment, Ideally I should regulate that to 0? 
- Set airspeed and climbrate
- **If I want to change airspeed and control altitude how does that work???**
  - I set an airspeed 
  - I need to set a pitch command that will set it to the desired altitude 

# What I need from Austin
- Give me the coefficients like usual 
- Give me the trim conditions, I'll handle the rest 
- 