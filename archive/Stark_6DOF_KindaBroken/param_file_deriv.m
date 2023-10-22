%% PTERA MDO Dynamic Simulation
%  Parameterization and Maneuver Definition

clc; clear; %close all;

dt = 0.01;
load plane_params.mat
PN = 254; % 8, 254, 255

%% Airfoil CL/alpha

% CLalpha = 'xfoil_naca0012_CLalpha.csv';
airfoil_data = readmatrix('xf-naca0012h-sa-1000000.csv');
uCLa = unique(airfoil_data,'rows');
Cla_max = max(uCLa(:,2));
alpha_vals = uCLa(:,1);
CLoindex = find(alpha_vals==0);
CL_plane = uCLa(:,2);
CLo = CL_plane(CLoindex);
CL_max = max(CL_plane);
% https://www.engineeringtoolbox.com/air-absolute-kinematic-viscosity-d_601.html
% http://airfoiltools.com/calculator/reynoldsnumber

% https://www.engineeringtoolbox.com/standard-atmosphere-d_604.html
altrho = 'alt_MSL_rho.csv';
density_data = readmatrix(altrho);
uAD = unique(density_data,'rows');
alt_vals = uAD(1:end-1,1); % MSL, meters
rho_vals = uAD(1:end-1,2); % kg/m3

%% User Input Aircraft Control Matrix

vCMD = plane_params(PN).performance.Voper;

% T = Time Length of Maneuver (seconds)
% V = Desired Speed for Maneuver (m/s)
% Psi = Desired Heading for Maneuver (rad)
[T, V, Psi, Alt] = deal(30, vCMD, 0, 50);
Steady1 = [T, V, Psi, Alt];

[T, V, Psi, Alt] = deal(30, vCMD, 0, 100);
Steady2 = [T, V, Psi, Alt];

[T, V, Psi, Alt] = deal(30, vCMD, 0, 200);
Steady3 = [T, V, Psi, Alt];

[T, V, Psi, Alt] = deal(30, vCMD, pi, 200);
Steady4 = [T, V, Psi, Alt];

[T, V, Psi, Alt] = deal(30, vCMD, pi, 100);
Steady5 = [T, V, Psi, Alt];

[T, V, Psi, Alt] = deal(30, vCMD, pi, 50);
Steady6 = [T, V, Psi, Alt];

[T, V, Psi, Alt] = deal(30, vCMD, 0, 50);
Steady8 = [T, V, Psi, Alt];

[T, V, Psi, Alt] = deal(30, vCMD, pi/2, 50);
Steady9 = [T, V, Psi, Alt];

% Ordering Maneuvers
% user_in = [Steady1; Steady2; Steady6; Steady7; Steady8; Steady9];
user_in = [Steady1; Steady2; Steady3; Steady4; Steady5; Steady6];

[Velocity_Cmd, Heading_Cmd, Altitude_Cmd, Ttime] = User_Input(dt,user_in);

altramp = 0; % 0 no, 1 yes
switch altramp
    case 0
    case 1
        % function [ramp_out] = ramp_input(alt_init,alt_fin,time_len,time_start,dt)
        alt_ramp = ramp_input(100,200,30,60,dt);
        Altitude_Cmd(6001:9000,2) = alt_ramp(:,2);

        % function [ramp_out] = ramp_input(alt_init,alt_fin,time_len,time_start,dt)
        alt_ramp = ramp_input(200,100,30,120,dt);
        Altitude_Cmd(12001:15000,2) = alt_ramp(:,2);

        % plot(Altitude_Cmd(:,1),Altitude_Cmd(:,2))
end

X0 = 0;
Y0 = 0;
Z0 = Altitude_Cmd(1,2);
V0 = Velocity_Cmd(1,2);

%% Aircraft Parameterization and Initial Conditions

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    c = plane_params(PN).geom_props.avg_chord; % tens
    b = plane_params(PN).geom_props.wingspan; % quarters
    
    % Mass, CDo, Ix, Iy
    mass = plane_params(PN).mass_props.mass;
    CDo = plane_params(PN).lon_derivs.CDo;
    CDa = 0.1; % NACA 2412
    Ix = plane_params(PN).mass_props.Ixx;
    Iy = plane_params(PN).mass_props.Iyy;
    Iz = plane_params(PN).mass_props.Izz;
    Ixz = plane_params(PN).mass_props.Ixz;
    
    Gamma = Ix*Iz - Ixz^2;
    c1 = ((Iy-Iz)*Iz - Ixz^2) / Gamma;
    c2 = ((Ix-Iy+Iz)*Ixz) / Gamma;
    c3 = Iz/Gamma; 
    c4 = Ixz/Gamma;
    c5 = (Iz-Ix)/Iy;
    c6 = Ixz/Iy;
    c7 = 1/Iy;
    c8 = Ix*(Ix-Iy) + Ixz^2;
    c9 = Ix/Gamma;
    
    % Velocity Angles
    CLa = abs(plane_params(PN).lon_derivs.CLalpha); %1/rad
    CYb = abs(plane_params(PN).lat_derivs.CYbeta); %1/rad
    Clb = abs(plane_params(PN).lat_derivs.Clbeta); %1/rad
    Cma = abs(plane_params(PN).lon_derivs.Cmalpha); %1/rad
    Cnb = abs(plane_params(PN).lat_derivs.Cnbeta); %1/rad

    % Rates
    CZq = abs(plane_params(PN).lon_derivs.CZq); %dimless
    CYp = abs(plane_params(PN).lat_derivs.CYp); %dimless
    CYr = abs(plane_params(PN).lat_derivs.CYr); %dimless
    Clp = abs(plane_params(PN).lat_derivs.Clp); %dimless
    Clr = abs(plane_params(PN).lat_derivs.Clr); %dimless
    Cmq = abs(plane_params(PN).lon_derivs.Cmq); %dimless
    Cnp = abs(plane_params(PN).lat_derivs.Cnp); %dimless
    Cnr = abs(plane_params(PN).lat_derivs.Cnr); %dimless

    % Surfaces
    CZde = abs(plane_params(PN).lon_derivs.CZde)*1/(pi/180); %1/deg
    CYda = abs(plane_params(PN).lat_derivs.CYda)*1/(pi/180); %1/deg
    CYdr = abs(plane_params(PN).lat_derivs.CYdr)*1/(pi/180); %1/deg
    Clda = abs(plane_params(PN).lat_derivs.Clda)*1/(pi/180); %1/deg
    Cldr = abs(plane_params(PN).lat_derivs.Cldr)*1/(pi/180); %1/deg
    Cmde = abs(plane_params(PN).lon_derivs.Cmde)*1/(pi/180); %1/deg
    Cnda = abs(plane_params(PN).lat_derivs.Cnda)*1/(pi/180); %1/deg
    Cndr = abs(plane_params(PN).lat_derivs.Cndr)*1/(pi/180); %1/deg
    
    % Conversions
    r2d = 180/pi;
    d2r = pi/180;
    
%% Weight, Thrust, G's

g = 9.81;
W = mass*g;
lbs = mass*2.20462;
power_max = plane_params(PN).components.PUSH_poweravail;
n_max = 10; % plane_params(PN).performance.StructGs;

%% Wing Shape

vstab_pos = 1;          % +1 Top, -1 Bottom
S = b*c;
K = plane_params(PN).geom_props.K;

%% Aircraft Lift Curves

% alpha_vals = (-15:0.25:20)';
% CL_plane = CLa*alpha_vals*(pi/180)+0.5;
% 
% if AR(row,col) >= 10 % magic number
%     % High AR
%     CLmax = 0.9*Cla_max; % Raymer pg. 405
%     % Saturating Curve
%     for i = 1:length(CL_plane)
%         if CL_plane(i) > CLmax
%             CL_plane(i) = CLmax;
%         end
%     end
% 
% else
%     % Low AR
%     CLmax = Cla_max*CLa/(2*pi);
%     % Saturating Curve
%     if CLmax > 0.9*Cla_max
%         CLmax = 0.9*Cla_max;
%     end
%     % Resaturate if scaled CLmax was above airfoil CLmax
%     for i = 1:length(CL_plane)
%         if CL_plane(i) > CLmax
%             CL_plane(i) = CLmax;
%         end
%     end
% 
% end

%% Gain set 1

find_k

%% Heading > Roll > Rate > da
Kp_psi_phi = 0.5; %0.5
Ki_psi_phi = 0;
Kd_psi_phi = 0;
Kp_phi_p = 1; %0.3
Ki_phi_p = 0.0;
Kd_phi_p = 0.0;
Kp_da_p = 0.03; % Klqr(1,4)/30; %
Ki_da_p = 0.0; %10
Kd_da_p = 0.0;
da_bounds = 25*pi/180;

%% V > Pitch > Rate > de
Kp_theta_V = 0.075;
Ki_theta_V = 0.2;
Kd_theta_V = 1.27;
Kp_q_theta = 10;
Ki_q_theta = 0.5;
Kd_q_theta = 0;
Kp_de_q = 0.05; % Klqr(2,5)/-72; % 
Ki_de_q = 0.1;
Kd_de_q = 0.03;
de_bounds = 25*pi/180;

%% Sideslip > Yaw > Rudder
Kp_beta_dr = 3; % Klqr(3,6); % 
Ki_beta_dr = 0.25;
Kd_beta_dr = 1;
dr_bounds = 25*pi/180;

%% Altitude > dH > Thrust
Kp_alt_dh = 1.5;
Ki_alt_dh = 0.3;
Kd_alt_dh = 1;
Kp_dh_t = 3; % mass/7.75; %
Ki_dh_t = 0.01;
Kd_dh_t = 0.01;

%% Old Gains

% %% Heading > Roll > Rate > da
% Kp_psi_phi = 2; %0.5
% Ki_psi_phi = 0;
% Kd_psi_phi = 0;
% Kp_phi_p = 1; %0.3
% Ki_phi_p = 0.0;
% Kd_phi_p = 0.0;
% Kp_da_p = 0.05; %0.03
% Ki_da_p = 0.0; %10
% Kd_da_p = 0.0;
% da_bounds = 25*pi/180;
% 
% %% V > Pitch > Rate > de
% Kp_theta_V = 0.075;
% Ki_theta_V = 0.2;
% Kd_theta_V = 1.27;
% Kp_q_theta = 10;
% Ki_q_theta = 0.5;
% Kd_q_theta = 0;
% Kp_de_q = 0.05;
% Ki_de_q = 0.1;
% Kd_de_q = 0.03;
% de_bounds = 25*pi/180;
% 
% %% Sideslip > Yaw > Rudder
% Kp_beta_r = 1;
% Ki_beta_r = 0.0;
% Kd_beta_r = 0.0;
% Kp_dr_r = 0.5;
% Ki_dr_r = 0;
% Kd_dr_r = 0;
% dr_bounds = 25*pi/180;
% 
% %% Altitude > dH > Thrust
% Kp_alt_dh = 3;
% Ki_alt_dh = 0.5;
% Kd_alt_dh = 0.0;
% Kp_dh_t = 5;
% Ki_dh_t = 0.0;
% Kd_dh_t = 0.01;
