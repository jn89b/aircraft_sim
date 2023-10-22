%% State-Space Analysis
clc; clear; close all;

%% User Input Aircraft Control Matrix

r2d = 180/pi;
d2r = pi/180;

% Body velocities
% +U = +X World (when aligned with world axes)
% +V = +Right World (when aligned with world axes)
% +W = +Z World (when aligned with world axes)
u = 0.5;
v = 0;
w = 0;

% +P = Clockwise about airplane longitudinal axis (right wing down)
% +Q = Nose down - I know the convention sucks but it works
%                - You can flip it but other things like Z break too
% +R = Clockwise about Z axis (to the right)

% This part just defines what rotational rates you want and for how long
% So for 10 seconds, no rotation
[T, p, q, r] = deal(10, 0, 0, 0);
Steady1 = [T, p, q, r];

% For 10 seconds, pitch rate of +9 deg/sec
[T, p, q, r] = deal(10, 0*d2r, 9*d2r, 0*d2r);
Steady2 = [T, p, q, r];

% For 10 seconds, pitch rate of -9 deg/sec
[T, p, q, r] = deal(10, 0*d2r, -9*d2r, 0*d2r);
Steady3 = [T, p, q, r];

% For 10 seconds, no rotation
[T, p, q, r] = deal(10, 0*d2r, 0*d2r, 0*d2r);
Steady4 = [T, p, q, r];

user_in = [Steady1; Steady2; Steady3; Steady4];

dt = 0.1;

[pCmd, qCmd, rCmd, Ttime] = User_Input(dt,user_in);

phi = zeros(Ttime/dt,1);
theta = zeros(Ttime/dt,1);
psi = zeros(Ttime/dt,1);
xpos = zeros(Ttime/dt,1);
ypos = zeros(Ttime/dt,1);
zpos = zeros(Ttime/dt,1);

%% Time Simulation

for iter = dt:dt:Ttime
    i = int32(iter/dt);
    
    %pqr2PhiDot_ThetaDot_PsiDot
    BRates = [pCmd(i,2); qCmd(i,2); rCmd(i,2)];
    BRates2ERates = [1 sin(phi(i))*tan(theta(i)) cos(phi(i))*tan(theta(i));
                     0 cos(phi(i))               -sin(phi(i));
                     0 sin(phi(i))*sec(theta(i)) cos(phi(i))*sec(theta(i))];
    Euler_Rates = BRates2ERates*BRates;
    
    dphi = Euler_Rates(1);
    dtheta = Euler_Rates(2);
    dpsi = Euler_Rates(3);
    
    % uvw2xyz
    BVels = [u; v; w];
    BVels2WVels = [cos(theta(i))*cos(psi(i)), sin(phi(i))*sin(theta(i))*cos(psi(i))-cos(phi(i))*sin(psi(i)), cos(phi(i))*sin(theta(i))*cos(psi(i))+sin(phi(i))*sin(psi(i));
                   cos(theta(i))*sin(psi(i)), sin(phi(i))*sin(theta(i))*sin(psi(i))+cos(phi(i))*cos(psi(i)), cos(phi(i))*sin(theta(i))*sin(psi(i))-sin(phi(i))*cos(psi(i));
                   -sin(theta(i)),             sin(phi(i))*cos(theta(i)),                                     cos(phi(i))*cos(theta(i))];
    WVels = BVels2WVels*BVels;
    
    dx = WVels(1);
    dy = WVels(2);
    dz = WVels(3);
    
    % Integration of World Pos/Angles
    xpos(i+1) = xpos(i) + dx*dt;
    ypos(i+1) = ypos(i) + dy*dt;
    zpos(i+1) = zpos(i) + dz*dt;
    
    phi(i+1) = phi(i) + dphi*dt;
    theta(i+1) = theta(i) + dtheta*dt;
    psi(i+1) = psi(i) + dpsi*dt;
    
end

time = pCmd(:,1);

%%

figure;

% Rotational Rates vs. Time
subplot(1,3,1)
plot(time,pCmd(:,2)*r2d,time,qCmd(:,2)*r2d,time,rCmd(:,2)*r2d);
legend('p','q','r')
grid on;
ylabel('Body Rates, deg/sec')
xlabel('Time, sec')

% World Angles vs. TIme
subplot(1,3,2)
plot(time,phi(1:end-1)*r2d)
hold on;
grid on;
ylim([-200 200])
plot(time,theta(1:end-1)*r2d)
plot(time,psi(1:end-1)*r2d)
legend('phi','theta','psi')
ylabel('World Angles, deg')
xlabel('Time, sec')

% World Positions vs. Time
subplot(1,3,3)
plot(time,xpos(1:end-1))
hold on;
grid on;
plot(time,ypos(1:end-1))
plot(time,zpos(1:end-1))
legend('xpos','ypos','zpos')
ylabel('World Positions')
xlabel('Time, sec')

%% 3D World Pos

figure;

plot3(xpos(:),ypos(:),zpos(:))
grid on;
hold on;
scatter3(0,0,0,24,'r','filled')
xlabel('X Position','Fontsize',15)
ylabel('Y Position','Fontsize',15)
zlabel('Z Position','Fontsize',15)
legend('Path','Start')
title({'Flight Path of Fixed-Wing Aircraft','in 6 Degrees of Freedom'},'Fontsize',15)
view(-45,20)

set(gca,'XDir','default')
set(gca,'YDir','reverse')
set(gca,'ZDir','default')

% xlim([-30 30])
% ylim([-15 15])
% zlim([-15 15])
