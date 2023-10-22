clc; clear; close all;

%% 6DOF

param_file_deriv

time = 0:dt:Ttime;
plottime = time(1:end-1);

psiCmd = Heading_Cmd(:,2);
altCmd = Altitude_Cmd(:,2);
VTCmd = Velocity_Cmd(:,2);

%%
alpha = [0 0];
beta  = [0 0];
dVT   = 0;
VT    = [V0 V0];
du    = 0;
dv    = 0;
dw    = 0;
u     = [V0 V0];
v     = [0 0];
w     = [0 0];
p     = [0 0];
q     = [0 0];
r     = [0 0];
phi   = [0 0];
theta = [0 0];
psi   = [0 0];
x     = [0 0];
y     = [0 0];
z     = [Z0 Z0];
dz    = 0;

VErrI     = [0 0];
thetaErrI = [0 0];
qErrI     = [0 0];
betaErrI  = [0 0];
rErrI     = [0 0];
AltErrI   = [0 0];
dhErrI    = [0 0];

states = [alpha(1);
          beta(1);
          VT(1);
          u(1);
          v(1);
          w(1);
          p(1);
          q(1);
          r(1);
          phi(1);
          theta(1);
          psi(1);
          x(1);
          y(1);
          z(1);
          1];

%%

for i = 1:length(time)
    
    if i > length(VTCmd)
        VTCmd(i) = VTCmd(i-1);
    end

% aerodynamics
rho = interp1(alt_vals,rho_vals,z(i));
Q   = 0.5*rho*VT(i)^2;
CL  = interp1(alpha_vals,CL_plane,alpha(i)*r2d);

% A
au = dw/(u(i)^2 + w(i)^2);
aw = -du/(u(i)^2 + w(i)^2);
bV = dv/(cos(beta(i))*u(i)^2);
bv = -dVT/(cos(beta(i))*u(i)^2);
Vu = du/VT(i);
Vv = dv/VT(i);
Vw = dw/VT(i);

XdT = 1/mass;
Xu = -(Q*S*CDo)/(mass*VT(i));
Xw = -(Q*S*CLo)/(mass*VT(i));
Xg = -g*sin(theta(i));
Xv = r(i);
Xq = -w(i);

YB = -(Q*S*CYb)/(mass);
Yv = YB/VT(i);
Yp = -(Q*S*b*CYp*vstab_pos)/(2*mass*VT(i));
Yr = (Q*S*b*CYr)/(2*mass*VT(i));
Yda = 0;
Ydr = -(Q*S*CYdr)/(mass);
Yg = g*sin(phi(i))*cos(theta(i));
Yu = -r(i);
Yw = p(i);

Zu = -(Q*S*CL)/(mass*VT(i));
Zw = -(Q*S)*(CDo+K*CL^2)/(mass*VT(i));
Zq = -(Q*S*c*CZq)/(2*mass*VT(i));
Zg = g*cos(phi(i))*cos(theta(i));
Zde = (Q*S*CZde)/(mass);
Zu = q(i) + Zu;
Zv = -p(i);

LB = c3*-(Q*S*b*Clb)/(Ix);
Lv = LB/VT(i);
Lp = c3*-(Q*S*b*b*Clp)/(2*Ix*VT(i));
Lr = c3*(Q*S*b*b*Clr)/(2*Ix*VT(i));
Lda = c3*(Q*S*b*Clda)/(Ix);
Ldr = c3*-(Q*S*b*Cldr)/(Ix);
% Lsum = c4*( LB+Lp+Lr+Lda+Ldr )/c3;
Lsum = 0;
Lq = c1*r(i) + c2*p(i);

Ma = c7*-(Q*S*c*Cma)/(Iy);
Mw = Ma/VT(i);
Mq = c7*-(Q*S*Cmq*c*c)/(2*VT(i)*Iy);
Mde = c7*(Q*S*c*Cmde)/(Iy);
Mp = c5*r(i) - c6*p(i);
Mr = c6*r(i);

NB = c9*(Q*S*Cnb*b)/(Iz);
Nv = NB/VT(i);
Np = c9*(Q*S*Cnp*b^2)/(2*VT(i)*Iz);
Nr = c9*-(Q*S*Cnr*b^2)/(2*VT(i)*Iz);
Nda = c9*-(Q*S*b*Cnda)/(Iz);
Ndr = c9*(Q*S*b*Cndr)/(Iz);
% Nsum = c4*(NB+Np+Nr+Nda+Ndr )/c9;
Nsum = 0;
Nq = c8*p(i) - c2*r(i);

dxu = cos(theta(i))*cos(psi(i));
dxv = sin(phi(i))*sin(theta(i))*cos(psi(i))-cos(phi(i))*sin(psi(i));
dxw = cos(phi(i))*sin(theta(i))*cos(psi(i))+sin(phi(i))*sin(psi(i));

dyu = cos(theta(i))*sin(psi(i));
dyv = sin(phi(i))*sin(theta(i))*sin(psi(i))+cos(phi(i))*cos(psi(i));
dyw = cos(phi(i))*sin(theta(i))*sin(psi(i))-sin(phi(i))*cos(psi(i));

dzu = sin(theta(i));
dzv = -sin(phi(i))*cos(theta(i));
dzw = -cos(phi(i))*cos(theta(i));

phip = 1;
phiq = sin(phi(i))*tan(theta(i));
phir = cos(phi(i))*tan(theta(i));

ttap = 0;
ttaq = cos(phi(i));
ttar = -sin(phi(i));

psip = 0;
psiq = sin(phi(i))/cos(theta(i));
psir = cos(phi(i))/cos(theta(i));

% STEVENS & LEWIS 81
% NELSON 123

% A MATRIX   a  B  VT  u   v   w    p    q    r  phi tta psi x  y  z  1
% Aalpha = [ 0  0  0  au   0  aw    0    0    0  0   0   0   0  0  0  0];
% Abeta  = [ 0  0 bV   0  bv   0    0    0    0  0   0   0   0  0  0  0];
% AVT    = [ 0  0  0  Vu  Vv  Vw    0    0    0  0   0   0   0  0  0  0];
% Au     = [ 0  0  0  Xu  Xv  Xw    0   Xq    0  0   0   0   0  0  0 Xg];
% Av     = [ 0  0  0  Yu  Yv  Yw   Yp    0   Yr  0   0   0   0  0  0 Yg];
% Aw     = [ 0  0  0  Zu  Zv  Zw    0   Zq    0  0   0   0   0  0  0 Zg];
% Ap     = [ 0  0  0   0  Lv   0   Lp    0   Lr  0   0   0   0  0  0 Nsum];
% Aq     = [ 0  0  0   0   0  Mw    0   Mq    0  0   0   0   0  0  0  0];
% Ar     = [ 0  0  0   0  Nv   0   Np    0   Nr  0   0   0   0  0  0 Lsum];
% Aphi   = [ 0  0  0   0   0   0 phip phiq phir  0   0   0   0  0  0  0];
% Atheta = [ 0  0  0   0   0   0 ttap ttaq ttar  0   0   0   0  0  0  0];
% Apsi   = [ 0  0  0   0   0   0 psip psiq psir  0   0   0   0  0  0  0];
% Ax     = [ 0  0  0 dxu dxv dxw    0    0    0  0   0   0   0  0  0  0];
% Ay     = [ 0  0  0 dyu dyv dyw    0    0    0  0   0   0   0  0  0  0];
% Az     = [ 0  0  0 dzu dzv dzw    0    0    0  0   0   0   0  0  0  0];
% A1     = [ 0  0  0   0   0   0    0    0    0  0   0   0   0  0  0  0];

Aalpha = [ 0  0  0  au   0  aw    0    0    0  0   0   0   0  0  0  0];
Abeta  = [ 0  0 bV   0  bv   0    0    0    0  0   0   0   0  0  0  0];
AVT    = [ 0  0  0  Vu  Vv  Vw    0    0    0  0   0   0   0  0  0  0];
Au     = [ 0  0  0  Xu  Xv  Xw    0   Xq    0  0   0   0   0  0  0 Xg];
Av     = [ 0  0  0  Yu  Yv  Yw   Yp    0   Yr  0   0   0   0  0  0 Yg];
Aw     = [ 0  0  0  Zu  Zv  Zw    0   Zq    0  0   0   0   0  0  0 Zg];
Ap     = [ 0  0  0   0  Lv   0   Lp    0   Lr  0   0   0   0  0  0 Nsum];
Aq     = [ 0  0  0   0   0  Mw    0   Mq    0  0   0   0   0  0  0  0];
Ar     = [ 0  0  0   0  Nv   0   Np    0   Nr  0   0   0   0  0  0 Lsum];
Aphi   = [ 0  0  0   0   0   0 phip phiq phir  0   0   0   0  0  0  0];
Atheta = [ 0  0  0   0   0   0 ttap ttaq ttar  0   0   0   0  0  0  0];
Apsi   = [ 0  0  0   0   0   0 psip psiq psir  0   0   0   0  0  0  0];
Ax     = [ 0  0  0 dxu dxv dxw    0    0    0  0   0   0   0  0  0  0];
Ay     = [ 0  0  0 dyu dyv dyw    0    0    0  0   0   0   0  0  0  0];
Az     = [ 0  0  0 dzu dzv dzw    0    0    0  0   0   0   0  0  0  0];
A1     = [ 0  0  0   0   0   0    0    0    0  0   0   0   0  0  0  0];

A = [Aalpha; Abeta; AVT; Au; Av; Aw; Ap; Aq; Ar; Aphi; Atheta; Apsi; Ax; Ay; Az; A1];

% B MATRIX
%          da  de  dr  dT
Balpha = [  0   0   0   0];
Bbeta  = [  0   0   0   0];
BVT    = [  0   0   0   0];
Bu     = [  0   0   0 XdT];
Bv     = [Yda   0 Ydr   0];
Bw     = [  0 Zde   0   0];
Bp     = [Lda   0 Ldr   0];
Bq     = [  0 Mde   0   0];
Br     = [Nda   0 Ndr   0];
Bphi   = [  0   0   0   0];
Btheta = [  0   0   0   0];
Bpsi   = [  0   0   0   0];
Bx     = [  0   0   0   0];
By     = [  0   0   0   0];
Bz     = [  0   0   0   0];
B1     = [  0   0   0   0];

B = [Balpha; Bbeta; BVT; Bu; Bv; Bw; Bp; Bq; Br; Bphi; Btheta; Bpsi; Bx; By; Bz; B1];

% B = zeros(16,4);

% % Velocity_Cmd, Heading_Cmd, Altitude_Cmd
% psiErr(i) = psiCmd(i-1) - psi(i);
% 
% phiCmd(i) = Kp_psi_phi*psiErr(i);
% 
%     Lmax = Q*S*CL_max;
%     aero = real(acos(W/(Lmax)));
%     aero(isnan(aero))=0;
%     struct = acos(1/n_max);
%     
%     if aero >= struct
%         Phi_max = struct;
%         Phi_min = -struct;
%     else
%         Phi_max = real(aero);
%         Phi_min = -real(aero);
%     end
%     
%     if phiCmd > Phi_max
%         phiCmd = Phi_max;
%     elseif phiCmd < Phi_min
%         phiCmd = Phi_min;
%     end
% 
% phiErr(i) = phiCmd(i) - phi(i);
% 
% pCmd(i) = Kp_phi_p*phiErr(i);
% pErr(i) = pCmd(i) - p(i);
% 
% da = Kp_da_p*pErr(i);
% if da > da_bounds
%     da = da_bounds;
% elseif da < da_bounds
%     da = -da_bounds;
% end
% 
% % Velocity_Cmd
% VErr(i+1) = -(VTCmd(i) - VT(i));
% VErrI(i+1) = VErrI(i) + VErr(i)*dt;
% VErrD(i+1) = ( VErr(i+1) - VErr(i) )/dt;
% 
% thetaCmd(i) = Kp_theta_V*(VErr(i+1) + Ki_theta_V*VErrI(i+1) + Kd_theta_V*VErrD(i+1));
% thetaErr(i+1) = thetaCmd(i) - theta(i);
% thetaErrI(i+1) = thetaErrI(i) + thetaErr(i)*dt;
% thetaErrD(i+1) = ( thetaErr(i+1) - thetaErr(i) )/dt;
% 
% qCmd(i) = Kp_q_theta*(thetaErr(i+1) + Ki_q_theta*thetaErrI(i+1) + Kd_q_theta*thetaErrD(i+1));
% qErr(i+1) = qCmd(i) - q(i);
% qErrI(i+1) = qErrI(i) + qErr(i)*dt;
% qErrD(i+1) = ( qErr(i+1) - qErr(i) )/dt;
% 
% de = Kp_de_q*(qErr(i+1) + Ki_de_q*qErrI(i+1) + Kd_de_q*qErrD(i+1));
% if de > de_bounds
%     de = de_bounds;
% elseif de < de_bounds
%     de = -de_bounds;
% end
% 
% % Beta_Cmd
% betaErr(i) = beta(i);
% betaErrI(i) = betaErrI(i-1) + betaErr(i)*dt;
% betaErrD(i) = ( betaErr(i) - betaErr(i-1) )/dt;
% 
% dr = Kp_beta_dr*betaErr(i) + Ki_beta_dr*betaErrI(i) + Kd_beta_dr*betaErrD(i);
% if dr > dr_bounds
%     dr = dr_bounds;
% elseif dr < dr_bounds
%     dr = -dr_bounds;
% end
%     
% % Altitude
% AltErr(i) = altCmd(i-1) - z(i);
% AltErrI(i) = AltErrI(i-1) + AltErr(i)*dt;
% AltErrD(i) = ( AltErr(i) - AltErr(i-1) )/dt;
% 
% dhCmd = Kp_alt_dh*AltErr(i) + Ki_alt_dh*AltErrI(i) + Kd_alt_dh*AltErrD(i);
% dhErr(i) = dhCmd - dz;
% dhErrI(i) = dhErrI(i-1) + dhErr(i)*dt;
% dhErrD(i) = ( dhErr(i) - dhErr(i-1) )/dt;
% 
% dT = Kp_dh_t*dhErr(i) + Ki_dh_t*dhErrI(i) + Kd_dh_t*dhErrD(i);
% Tmax = power_max/VT(i);
% if dT > Tmax
%     dT = Tmax;
% elseif dT < 0
%     dT = 0;
% end

da = 0;
de = 0;
dr = 0;
dT = 0;

if da > da_bounds
    da = da_bounds;
elseif dT < -da_bounds
    dT = -da_bounds;
end

if de > de_bounds
    de = de_bounds;
elseif de < -de_bounds
    de = -de_bounds;
end

if dr > dr_bounds
    dr = dr_bounds;
elseif dr < -dr_bounds
    dr = -dr_bounds;
end

if dT > 50
    dT = 50;
elseif dT < 0
    dT = 0;
end

inputs = [da; de; dr; dT];

% dX = AX + BU
dstates = A*states + B*inputs;

dalpha = dstates(1);
dbeta  = dstates(2);
dVT    = dstates(3);
du     = dstates(4);
dv     = dstates(5);
dw     = dstates(6);
dp     = dstates(7);
dq     = dstates(8);
dr     = dstates(9);
dphi   = dstates(10);
dtheta = dstates(11);
dpsi   = dstates(12);
dx     = dstates(13);
dy     = dstates(14);
dz     = dstates(15);

% INTEGRATE STATES
alpha(i+1) = alpha(i) + dalpha*dt;
beta(i+1)  = beta(i) + dbeta*dt;
VT(i+1)    = VT(i) + dVT*dt;
u(i+1)     = u(i) + du*dt;
v(i+1)     = v(i) + dv*dt;
w(i+1)     = w(i) + dw*dt;
p(i+1)     = p(i) + dp*dt;
q(i+1)     = q(i) + dq*dt;
r(i+1)     = r(i) + dr*dt;
phi(i+1)   = phi(i) + dphi*dt;
theta(i+1) = theta(i) + dtheta*dt;
psi(i+1)   = psi(i) + dpsi*dt;
x(i+1)     = x(i) + dx*dt;
y(i+1)     = y(i) + dy*dt;
z(i+1)     = z(i) + dz*dt;

states = [alpha(i+1);
          beta(i+1);
          VT(i+1);
          u(i+1);
          v(i+1);
          w(i+1);
          p(i+1);
          q(i+1);
          r(i+1);
          phi(i+1);
          theta(i+1);
          psi(i+1);
          x(i+1);
          y(i+1);
          z(i+1);
          1];

save4later(i).Astuff = (states').*A;
save4later(i).Bstuff = (inputs').*B;
save4later(i).CL = CL;
save4later(i).du = du;
save4later(i).dv = dv;
save4later(i).dw = dw;
save4later(i).da = da;
save4later(i).de = de;
save4later(i).dr = dr;
save4later(i).dT = dT;

statesave(i).states = states;

dstatesave(i).dstates = dstates;

end

%%

for j = 1:length(save4later)
    uforces(j,:) = save4later(j).Astuff(4,:);
    vforces(j,:) = save4later(j).Astuff(5,:);
    wforces(j,:) = save4later(j).Astuff(6,:);
    
    uforcesB(j,:) = save4later(j).Bstuff(4,:);
    vforcesB(j,:) = save4later(j).Bstuff(5,:);
    wforcesB(j,:) = save4later(j).Bstuff(6,:);
    
    statelist(j,:) = statesave(j).states';
    dstatelist(j,:) = dstatesave(j).dstates';
    CL(j,:) = save4later(j).CL;
    du(j,:) = save4later(j).du;
    dv(j,:) = save4later(j).dv;
    dw(j,:) = save4later(j).dw;
    da(j,:) = save4later(j).da;
    de(j,:) = save4later(j).de;
    dr(j,:) = save4later(j).dr;
    dT(j,:) = save4later(j).dT;
end

figure(1)
hold on; grid on;
plot(time,u(2:end))
plot(time,v(2:end))
plot(time,w(2:end))
plot(time,VT(2:end))
xlabel('times, seconds')
ylabel('meters per second')
legend('u','v','w','VT')

figure(2)
hold on; grid on;
plot(time,p(2:end)*r2d)
plot(time,q(2:end)*r2d)
plot(time,r(2:end)*r2d)
xlabel('times, seconds')
ylabel('degrees per second')
legend('p','q','r')

figure(3)
hold on; grid on;
plot(time,phi(2:end)*r2d)
plot(time,theta(2:end)*r2d)
plot(time,psi(2:end)*r2d)
xlabel('times, seconds')
ylabel('world degrees')
legend('phi','theta','psi')

figure(4)
hold on; grid on;
plot(time,x(2:end))
plot(time,y(2:end))
plot(time,z(2:end))
xlabel('time, seconds')
ylabel('position, m')
legend('x','y','z')

figure(5)
subplot(2,1,1)
hold on; grid on;
plot(time,alpha(2:end)*r2d)
plot(time,beta(2:end)*r2d)
xlabel('time, seconds')
ylabel('degrees')
legend('alpha','beta')

subplot(2,1,2)
hold on; grid on;
plot(time,CL)
xlabel('time, seconds')
ylabel('CL')

figure(6)
subplot(1,3,1)
hold on; grid on;
plot(time,uforces(:,4)) %u
plot(time,uforces(:,5)) %v
plot(time,uforces(:,6)) %w
plot(time,uforces(:,7)) %p
plot(time,uforces(:,8)) %q
plot(time,uforces(:,9)) %r
plot(time,uforces(:,end)) %g
plot(time,uforcesB(:,2)) %de
plot(time,uforcesB(:,4)) %dT
legend('u','v','w','p','q','r','g','de','dT')
xlabel('seconds')
title('u forces')

subplot(1,3,2)
hold on; grid on;
plot(time,vforces(:,4)) %u
plot(time,vforces(:,5)) %v
plot(time,vforces(:,6)) %w
plot(time,vforces(:,7)) %p
plot(time,vforces(:,8)) %q
plot(time,vforces(:,9)) %r
plot(time,vforces(:,end)) %g
plot(time,vforcesB(:,1)) %da
plot(time,vforcesB(:,3)) %dr
legend('u','v','w','p','q','r','g','da','dr')
xlabel('seconds')
title('v forces')

subplot(1,3,3)
hold on; grid on;
plot(time,wforces(:,4)) %u
plot(time,wforces(:,5)) %v
plot(time,wforces(:,6)) %w
plot(time,wforces(:,7)) %p
plot(time,wforces(:,8)) %q
plot(time,wforces(:,9)) %r
plot(time,wforces(:,end)) %g
plot(time,wforcesB(:,2)) %de
plot(time,wforcesB(:,4)) %dT
legend('u','v','w','p','q','r','g','de','dT')
xlabel('seconds')
title('w forces')

% figure(7)
% hold on; grid on;
% plot(time,de*r2d) %u
% 
% figure(8)
% subplot(1,3,1)
% hold on; grid on;
% plot(VErr)
% plot(VErrI)
% plot(VErrD)
% title('V errs')
% legend('P','I','D')
% 
% subplot(1,3,2)
% hold on; grid on;
% plot(thetaErr)
% plot(thetaErrI)
% plot(thetaErrD)
% title('theta errs')
% legend('P','I','D')
% 
% subplot(1,3,3)
% hold on; grid on;
% plot(qErr)
% plot(qErrI)
% plot(qErrD)
% title('q errs')
% legend('P','I','D')
