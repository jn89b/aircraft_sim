function [Velocity_Cmd, Heading_Cmd, Altitude_Cmd, Total_Time] = User_Input(dt,user_in)

%% Desired Speeds, Headings and Timings

Total_Time = sum(user_in(:,1));
N_points = Total_Time/dt;
endT = zeros(N_points,1);

for i = 1:N_points
    endT(i) = i*dt;
end

%% Initialize Matrices

Velocity_Cmd = zeros(N_points,2);
Heading_Cmd = zeros(N_points,2);
Altitude_Cmd = zeros(N_points,2);

Velocity_Cmd(:,1) = (endT);
Heading_Cmd(:,1) = (endT);
Altitude_Cmd(:,1) = (endT);

ts_new = 0;
ts_old = 0;

user_len = size(user_in);

for k = 1:user_len(1)
    
    ts_new = user_in(k,1)/dt;
    
    for j = 1:ts_new
        
        Velocity_Cmd(ts_old+j,2) = user_in(k,2);
        Heading_Cmd(ts_old+j,2) = user_in(k,3);
        Altitude_Cmd(ts_old+j,2) = user_in(k,4);
        
    end
    
    ts_old = ts_old + ts_new;
    
end