% state init
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
theta = [15*d2r 15*d2r];
psi   = [0 0];
x     = [0 0];
y     = [0 0];
z     = [Z0 Z0];
dz    = 0;

states = [u(1);
          v(1);
          w(1);
          p(1);
          q(1);
          r(1);
          phi(1);
          theta(1);
          psi(1);
          z(1)];

% aerodynamics
rho = interp1(alt_vals,rho_vals,z(1));
Q   = 0.5*rho*VT(1)^2;
CL  = interp1(alpha_vals,CL_plane,alpha(1)*r2d);

% A
au = dw/(u(1)^2 + w(1)^2);
aw = -du/(u(1)^2 + w(1)^2);
bV = dv/(cos(beta(1))*u(1)^2);
bv = -dVT/(cos(beta(1))*u(1)^2);
Vu = du/VT(1);
Vv = dv/VT(1);
Vw = dw/VT(1);

XdT = 1/mass;
Xu = -(Q*S*CDo)/(mass*VT(1));
Xw = -(Q*S*CLo)/(mass*VT(1));
Xg = -g*sin(theta(1));
Xv = r(1);
Xq = -w(1);

YB = -(Q*S*CYb)/(mass);
Yv = YB/VT(1);
Yp = -(Q*S*b*CYp*vstab_pos)/(2*mass*VT(1));
Yr = (Q*S*b*CYr)/(2*mass*VT(1));
Yda = 0;
Ydr = -(Q*S*CYdr)/(mass);
Yg = g*sin(phi(1))*cos(theta(1));
Yu = -r(1);
Yw = p(1);

Zu = -(Q*S*CL)/(mass*VT(1));
Zw = -(Q*S)*(CDo+K*CL^2)/(mass*VT(1));
Zq = -(Q*S*c*CZq)/(2*mass*VT(1));
Zg = g*cos(phi(1))*cos(theta(1));
Zde = (Q*S*CZde)/(mass);
Zu = q(1) + Zu;
Zv = -p(1);

LB = c3*-(Q*S*b*Clb)/(Ix);
Lv = LB/VT(1);
Lp = c3*-(Q*S*b*b*Clp)/(2*Ix*VT(1));
Lr = c3*(Q*S*b*b*Clr)/(2*Ix*VT(1));
Lda = c3*(Q*S*b*Clda)/(Ix);
Ldr = c3*-(Q*S*b*Cldr)/(Ix);
Lsum = 0; % c4*( LB+Lp+Lr+Lda+Ldr )/c3;
Lq = c1*r(1) + c2*p(1);

Ma = c7*-(Q*S*c*Cma)/(Iy);
Mw = Ma/VT(1);
Mq = c7*-(Q*S*Cmq*c*c)/(2*VT(1)*Iy);
Mde = c7*(Q*S*c*Cmde)/(Iy);
Mp = c5*r(1) - c6*p(1);
Mr = c6*r(1);

NB = c9*(Q*S*Cnb*b)/(Iz);
Nv = NB/VT(1);
Np = c9*(Q*S*Cnp*b^2)/(2*VT(1)*Iz);
Nr = c9*-(Q*S*Cnr*b^2)/(2*VT(1)*Iz);
Nda = c9*-(Q*S*b*Cnda)/(Iz);
Ndr = c9*(Q*S*b*Cndr)/(Iz);
Nsum = 0; % c4*(NB+Np+Nr+Nda+Ndr )/c9;
Nq = c8*p(1) - c2*r(1);

dxu = cos(theta(1))*cos(psi(1));
dxv = sin(phi(1))*sin(theta(1))*cos(psi(1))-cos(phi(1))*sin(psi(1));
dxw = cos(phi(1))*sin(theta(1))*cos(psi(1))+sin(phi(1))*sin(psi(1));

dyu = cos(theta(1))*sin(psi(1));
dyv = sin(phi(1))*sin(theta(1))*sin(psi(1))+cos(phi(1))*cos(psi(1));
dyw = cos(phi(1))*sin(theta(1))*sin(psi(1))-sin(phi(1))*cos(psi(1));

dzu = sin(theta(1));
dzv = -sin(phi(1))*cos(theta(1));
dzw = -cos(phi(1))*cos(theta(1));

phip = 1;
phiq = sin(phi(1))*tan(theta(1));
phir = cos(phi(1))*tan(theta(1));

ttap = 0;
ttaq = cos(phi(1));
ttar = -sin(phi(1));

psip = 0;
psiq = sin(phi(1))/cos(theta(1));
psir = cos(phi(1))/cos(theta(1));

%% A Matrix
%           u   v   w    p    q    r phi tta z
Au     = [ Xu  Xv  Xw    0   Xq    0   0   0 0]; % -g*sin(theta)
Av     = [ Yu  Yv  Yw   Yp    0   Yr   0   0 0]; % +g*cos(theta)*sin(phi)
Aw     = [ Zu  Zv  Zw    0   Zq    0   0   0 0]; % +g*cos(theta)*cos(phi)
Ap     = [  0  Lv   0   Lp    0   Lr   0   0 0];
Aq     = [  0   0  Mw    0   Mq    0   0   0 0];
Ar     = [  0  Nv   0   Np    0   Nr   0   0 0];
Aphi   = [  0   0   0 phip phiq phir   0   0 0];
Atheta = [  0   0   0 ttap ttaq ttar   0   0 0];
Az     = [dzu dzv dzw    0    0    0   0   0 0];

Amatrix = [Au; Av; Aw; Ap; Aq; Ar; Aphi; Atheta; Az]
numstates = length(Amatrix);

% B MATRIX
%          da  de  dr  dT
Bu     = [  0   0   0 XdT];
Bv     = [Yda   0 Ydr   0];
Bw     = [  0 Zde   0   0];
Bp     = [Lda   0 Ldr   0];
Bq     = [  0 Mde   0   0];
Br     = [Nda   0 Ndr   0]; 
Bphi   = [  0   0   0   0];
Btheta = [  0   0   0   0];
Bz     = [  0   0   0   0];

Bmatrix = [Bu; Bv; Bw; Bp; Bq; Br; Bphi; Btheta; Bz];
numctrls = size(Bmatrix,2);

%% C Matrix
Cmatrix = eye(numstates);

%% D Matrix
Dmatrix = zeros(numstates,numctrls);

%%

Q = eye(numstates);
R = eye(numctrls);

sys = ss(Amatrix,Bmatrix,Cmatrix,Dmatrix);
[Klqr,Slqr,Plqr] = lqr(sys,Q,R)

xu   = [ V0 V0];
xv   = [  0  0];
xw   = [  0  0];
xp   = [  0  0];
xq   = [  0  0];
xr   = [  0  0];
xphi = [  0  0];
xtta = [0.1  0];
xpsi = [  0  0];
xz   = [ Z0 Z0];

xcmds = [xu; xv; xw; xp; xq; xr; xphi; xtta; xpsi; xz];
x0 = xcmds(:,1);
xd = xcmds(:,2);

%% LQR
