    # -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 18:11:48 2022
 
@author: jnguy
"""
 
import numpy as np
import pandas as pd
import os
import glob
import math
import util_functions
 
import matplotlib.pyplot as plt
import control
 
import plot_hw
 
from scipy import signal
 
class Aircraft(object):
    """define the attributes and parameters of aircraft
    inputs must be a pandas series format
    """
 
 
    def __init__(self, phys_df, lon_df, lat_df):
        self.physical_data = phys_df
        self.lon_data = lon_df
        self.lat_data = lat_df
       
        self.C_l = lon_df['C_l']
        self.C_d = lon_df['C_d']
        self.C_da = lon_df['C_da']
        self.C_la = lon_df['C_la']
        self.C_lde = lon_df['C_lde']
        self.C_mq = lon_df['C_mq']
        self.C_ma = lon_df['C_ma']
        self.C_du = 0.0
       
        self.S = phys_df['S-m2']
        self.Iy = phys_df['Iy-kgm^2']
        self.Ix = phys_df['Ix-kgm^2']
        self.Iz = phys_df['Iz-kgm^2']
        self.mass = phys_df['Weight-kg']
        self.c = phys_df['c_-m']
        self.b = phys_df['b-m']
       
        self.C_lp = lat_df['C_lp']
        self.C_yb = lat_df['C_ybeta']
        self.C_nbeta = lat_df['C_nbeta']
        self.C_lbeta = lat_df['C_lbeta']
        self.C_lr = lat_df['C_lr']
        self.C_nr = lat_df['C_nr']
        self.C_np = lat_df['C_np']
       
        #rudder
        self.C_ydeltar = lat_df['C_ydeltar']
        self.C_ldeltar = lat_df['C_ldeltar']
       
        self.C_ndeltaa = lat_df['C_ndeltaa']
        self.C_ndeltar = lat_df['C_ndeltar']
        self.C_ldeltaa = lat_df['C_ldeltaa']
       
        self.C_mdeltae = lon_df['C_mbe']
 
    def compute_roll_damping(
            self, velocity_array, C_lp=None, Ix=None):
        """Compute L_p roll damping from velocity array should be a
        linear response
        rad/s^2
        """
        if C_lp is None:
            C_lp  = self.lat_data['C_lp']
       
        if Ix is None:
            Ix = self.physical_data['Ix-kgm^2']
           
        S = self.physical_data['S-m2']
        b = self.physical_data['b-m']
   
        #dynamic pressure, rho
        rho = 1.204 #kg/m^2
        q = 0.5*velocity_array**2*rho
        L_p = ((S*(b**2)*C_lp)/(2*Ix))*(q/velocity_array)
       
        return L_p
   
    def compute_roll_damping_aileron(self, velocity_array, C_ldelta=None,
                                     Ix=None):
        """compute the L_delta_a , should be a quadratic response
        units are rad/s^2"""
        if C_ldelta is None:
            C_ldelta  = self.lat_data['C_ldeltaa']
           
        if Ix is None:
            Ix = self.physical_data['Ix-kgm^2']
           
        S = self.physical_data['S-m2']
        b = self.physical_data['b-m']
        Ix = self.physical_data['Ix-kgm^2']
       
        #dynamic pressure, rho
        rho = 1.204 #kg/m^2
        q = 0.5*velocity_array**2*rho
        L_da = (q*S*b*C_ldelta)/Ix
       
        return L_da
   
    def compute_maximum_roll_rate(self, delta_a, velocity_array, L_p, L_da):
        """compute maximum roll rate of aircraft, alpha is angle of attack in
        degrees"""
       
        return (-L_da/ L_p) * delta_a
   
    def compute_open_roll_response(self, delta_a, velocity_array):
        """compute roll acceleration based on velocity"""
        L_da = self.compute_roll_damping_aileron(velocity_array)
        L_p = self.compute_roll_damping(velocity_array)
        p = self.compute_maximum_roll_rate(delta_a, velocity_array,L_p, L_da)
       
        p_dot = (L_p*p) + (L_da*delta_a)
        return p_dot
   
    def fail_aileron(self, percent_bnds):
        """fail aileron based on failure bound array non dimensionless val
        takes in a percentage decimal bound and returns a list of the c_ldelta
        of the wing left over"""
        C_ldelta  = self.lat_data['C_ldeltaa']
        c_ldelta_fail_list = []
        for percent in percent_bnds:
            #one side full + percentage of remainder
            split_C_ldelta = (C_ldelta/2)+ ((C_ldelta/2)*percent) # split in half multiply
            c_ldelta_fail_list.append(split_C_ldelta)
           
        return c_ldelta_fail_list
   
    def compute_L_deltar(self,q):
        """compute """
        return (q * self.S * self.b * self.C_ldeltar) / (self.Ix)
   
    def compute_Lp(self,q, velocity):
        """compute L_p, roll damping from velocity"""
        L_p = ((self.S*(self.b**2)*self.C_lp)/(2*self.Ix))*(q/velocity)
 
        return L_p
   
    def compute_Lda(self,q, velocity):
        """compute L_da"""
        return (q * self.S * self.b * self.C_ldeltaa) / (self.Ix)
   
    def compute_L_beta(self,q):
        """computes L_beta from dynamic pressure, q"""
        return (q * self.S * self.b * self.C_lbeta)/ self.Iy
   
    def compute_L_p(self,q,velocity):
        """computes L_p """
        return (q * self.S * self.b**2 * self.C_lp)/ (2 * self.Ix * velocity)
 
    def compute_L_r(self,q,velocity):
        """computes L_p """
        return (q * self.S * self.b**2 * self.C_lr)/ (2 * self.Ix * velocity)
 
    def compute_M_delta_e(self, q):        
        """computes the M_alpha or change of moment from the angle of attack"""
        return (self.C_mdeltae* q * self.S * self.c)/self.Iy  
   
    def compute_M_alpha(self, q, velocity):        
        """computes the M_alpha or change of moment from the angle of attack"""
        return velocity * self.compute_M_w(q,velocity)
   
    def compute_M_q(self, q, velocity):        
        """computes the M_q or change of moment from the change of pitch rate
        defaults to standard physical parameters of aircraft"""
        #return self.C_mq * (self.c/2*velocity) * ((q * self.S * self.c)/self.Iy)
        return (self.C_mq * self.c**2 * q * self.S)/ (2 * velocity * self.Iy)
 
    def compute_M_u(self):        
        """computes the M_q or change of moment from the change of pitch rate
        defaults to standard physical parameters of aircraft"""
        return 0.0
   
    def compute_M_w(self,q,velocity):
        return (self.C_ma * q * self.S * self.c)/(self.Iy * velocity)  
 
    def compute_N_beta(self,q):
        """computes N_beta from dynamic presssure,q,"""
        return (q * self.S * self.b * self.C_nbeta)/ self.Iz
   
    def compute_N_p(self,q, velocity):
        """computes N_p from q and row"""
        return (q * self.S * self.b**2 * self.C_np)/ (2 * self.Iz * velocity)
   
    def compute_N_r(self,q, velocity):
        """compute N_r"""
        return (q * self.S * self.b**2 * self.C_nr)/ (2 * self.Iz * velocity)
   
    def compute_N_deltaa(self, q):
        """"""
        return -(q * self.S * self.b * self.C_ndeltaa)/self.Iz
 
    def compute_N_deltar(self,q):
        """"""
        return (q * self.S * self.b * self.C_ndeltar)/self.Iz
 
    def compute_X_u(self, q, velocity):
        return -((0 + (2 * self.C_d)) * q * velocity)/ (self.mass * velocity)
   
    def compute_X_w(self, q, velocity):
        return -((self.C_da - self.C_l) * q * velocity)/ (self.mass * velocity)
 
    def compute_X_delta_e(self, q):
        """computes the Z_alpha or the chnge of Z force with change of
        angle of attack"""
        return (-self.C_lde * q * self.S)/ (self.mass)
   
    def compute_Y_beta(self,q):
        """computes Y_beta from dynamic presssure,q,"""
        return (q * self.S * self.C_yb)/ self.mass
   
    def compute_Y_p(self):
        """set this to 0"""
        return 0.0
   
    def compute_Y_r(self,q, velocity):
        """computes Y_r from dynamic pressure, q and from velocity"""
        return (q * self.S * self.b * self.C_ydeltar) / (2 * self.mass * velocity)
   
    #https://stackoverflow.com/questions/55497837/how-to-tell-a-function-to-use-the-default-argument-values
    # def compute_Z_W(self, alpha, q, velocity):
    #     """computes the Z_alpha or the chnge of Z force with change of
    #     angle of attack"""
    #     return ((self.C_la + self.C_d) * q * self.S)/(velocity * self.mass)
   
    def compute_Z_delta_e(self, q):
        """computes the Z_alpha or the chnge of Z force with change of
        angle of attack"""
        return -(self.C_lde* q * self.S)/ (self.mass)
        #return -((self.C_la + self.C_d) * q * self.S)/(velocity*self.mass)
 
    def compute_Z_u(self, q, velocity):
        return -((0.0 - (2 * self.C_l)) * q * self.S)/(velocity * self.mass)
       
    def compute_Z_w(self, q, velocity):
        print(-((self.C_la + self.C_d) * q * self.S)/(velocity * self.mass))
        return -((self.C_la + self.C_d) * q * self.S)/(velocity * self.mass)
       
class AircraftStateSpace():
    """
    state space
    """
    def __init__(self, aircraft):
        self.aircraft = aircraft
 
    def compute_roll_mode(self, q, velocity, input_cmd):
        """compute first order roll mode, returns t and  p, time
        of reponse and roll_rate, p_rate"""
        L_p = self.aircraft.compute_Lp(q, velocity)
        L_da = self.aircraft.compute_Lda(q, velocity)
        system = self.compute_state_space(L_p, L_da, input_cmd, 0)
 
        return system
       
        # t_p_rate, p_rate = signal.step(system)
        # S = control.ss(np.array(L_p),np.array(L_da),1,0)
        # tf = control.ss2tf(S)
        # control.rlocus(tf)
       
        # return L_p
   
    def compute_short_period_mode(self, alpha, q, velocity):
        """get 2nd order response of short period mode
        q is pitch axis rate
        """
        Z_alpha =  self.aircraft.compute_Z_w( q, velocity)
        Z_deltae = self.aircraft.compute_Z_delta_e(q)
       
        M_alpha = self.aircraft.compute_M_alpha(q, velocity)
        M_q = self.aircraft.compute_M_q(q, velocity)
        M_de = self.aircraft.compute_M_delta_e(q)
       
 
        A = [[Z_alpha, 1],
             [M_alpha, M_q]]
       
   
        B = [[Z_deltae/velocity],
              [M_de]]
 
        print("a", A)
        print("b", B)
 
        C = [1, 0]
       
        D = [0.0]
       
        system = self.compute_state_space(A, B, C, D)
        #G = cntrl.rlocus(system)
 
        return system
       
        # system = signal.ss2tf(A, B, C, D)
        # t,r = signal.step(system)
       
        # t = np.linspace(t[0], t[-1], num=len(t))
        # u = np.ones(len(t))
        # t,y,r = signal.lsim(system, u, t)
       
        # S = control.ss(np.array(A),np.array(B),C,D)
        # tf = control.ss2tf(S)
        # print(tf)
       
        # return A
       
    def compute_dutch_roll_mode(self, alpha, q, velocity, delta_r):
        """get 3rd order dutch roll response"""
        #COMPUTE values for side slip
        Y_beta = self.aircraft.compute_Y_beta(q)
        Y_p = self.aircraft.compute_Y_p()
        Y_r = self.aircraft.compute_Y_r(q, velocity)
        #g_cos = (9.81/velocity) * math.cos(theta)
       
        # compute values for roll
        L_beta = self.aircraft.compute_L_beta(q)
        L_p = self.aircraft.compute_L_p(q, velocity)
        L_r = self.aircraft.compute_L_r(q, velocity)
       
        # compute valueass for  pitch
        N_beta = self.aircraft.compute_N_beta(q)
        N_p = self.aircraft.compute_N_p(q, velocity)
        N_r = self.aircraft.compute_N_r(q,velocity)
 
        # Inputs
        L_da = self.aircraft.compute_Lda(q, velocity)
        L_dr = self.aircraft.compute_L_deltar(q)
        N_da = self.aircraft.compute_N_deltaa(q)
        N_dr = self.aircraft.compute_N_deltar(q)
       
        A = [[Y_beta/velocity,  Y_p/velocity,   -(1 - (Y_r/velocity))],
              [L_beta,           L_p,            L_r],
              [N_beta,           N_p,            N_r]]
       
       
        B = [[0.0],
              [L_da],
              [N_da]]
 
        C = [[1.0,    1.0,   1.0]]
       
        # n = number of inputs
        # p = single
        # q = outputs of state so three outputs is size 3
        #C = q x n - >
        #D = q x p # 3 x 1
        D = [0.0]
       
        system = self.compute_state_space(A, B, C, D)
        t, r = signal.step(system)
       
        t = np.linspace(t[0], t[-1], num=len(t))
        u = np.ones(len(t))*0.529
        t,y,r = signal.lsim(system, u, t)
 
        S = control.ss(np.array(A),np.array(B),C,D)
        tf = control.ss2tf(S)
        control.rlocus(tf)
       
        return A
   
    def compute_long_dynamics(self, alpha, q, velocity):
        X_u = self.aircraft.compute_X_u(q, velocity)
        X_w = self.aircraft.compute_X_w(q, velocity)
        Z_u = self.aircraft.compute_Z_u(q, velocity)
        Z_w = self.aircraft.compute_Z_w(q, velocity)
        M_w = self.aircraft.compute_M_w(q, velocity)
        M_q = self.aircraft.compute_M_q(q, velocity)
       
        X_d = self.aircraft.compute_X_delta_e(q)
        Z_d = self.aircraft.compute_Z_delta_e(q)
        M_d = self.aircraft.compute_M_delta_e(q)
           
   
        A = [[X_u,  X_w,    0,        9.81], #delta u
             [Z_u,  Z_w,    velocity, 0], #delta w
             [0,    M_w,    M_q,      0], #delta q
             [0,    0,      0,        1]] # delta theta
   
        B = [[0],
             [Z_d],
             [M_d],
             [0]]
 
        C = [1, 1, 1, 1]
       
        D = [0.0]
       
        system = self.compute_state_space(A, B, C, D)
        system = signal.ss2tf(A, B, C, D)
        #G = cntrl.rlocus(system)
       
        t,r = signal.step(system)
       
        t = np.linspace(t[0], t[-1], num=len(t))
        u = np.ones(len(t))
        t,y,r = signal.lsim(system, u, t)
       
        S = control.ss(np.array(A),np.array(B),C,D)
        tf = control.ss2tf(S)
        #control.rlocus(tf)
       
        return A
   
    def compute_lat_dynamics(self, alpha,q,velocity):
        """"""
        #COMPUTE values for side slip
        Y_beta = self.aircraft.compute_Y_beta(q)
        Y_p = self.aircraft.compute_Y_p()
        Y_r = self.aircraft.compute_Y_r(q, velocity)
        #g_cos = (9.81/velocity) * math.cos(theta)
       
        # compute values for roll
        L_beta = self.aircraft.compute_L_beta(q)
        L_p = self.aircraft.compute_L_p(q, velocity)
        L_r = self.aircraft.compute_L_r(q, velocity)
       
        # compute valueass for  pitch
        N_beta = self.aircraft.compute_N_beta(q)
        N_p = self.aircraft.compute_N_p(q, velocity)
        N_r = self.aircraft.compute_N_r(q,velocity)
 
        # Inputs
        L_da = self.aircraft.compute_Lda(q, velocity)
        L_dr = self.aircraft.compute_L_deltar(q)
        N_da = self.aircraft.compute_N_deltaa(q)
        N_dr = self.aircraft.compute_N_deltar(q)
       
        A = [[Y_beta/velocity,  Y_p/velocity,   -(1 - (Y_r/velocity)),      9.81/velocity],
              [L_beta,           L_p,            L_r,                           0],
              [N_beta,           N_p,            N_r,                           0],
              [0,                 1,             0,                             0]]
 
        B = [[0.0],
              [L_da],
              [N_da],
              [0.0]]
 
        C = [[1.0,    1.0,   1.0,   1.0]]
       
        # n = number of inputs
        # p = single
        # q = outputs of state so three outputs is size 3
        #C = q x n - >
        #D = q x p # 3 x 1
        D = [[0.0]]
       
        system = self.compute_state_space(A, B, C, D)
        t, r = signal.step(system)
       
        t = np.linspace(t[0], t[-1], num=len(t))
        u = np.ones(len(t))*0.529
        t,y,r = signal.lsim(system, u, t)
 
        S = control.ss(np.array(A),np.array(B),C,D)
        tf = control.ss2tf(S)
        #control.rlocus(tf)
        #z,p, k = signal.tf2zpk(num, den)    
        return A
   
    def compute_state_space(self,A,B,C,D):
        """returns the system space space as signal.StateSpace format"""
        # c is a measurement matrix typically identity matrix
        a = np.array(B)
        return signal.StateSpace(A,B,C,D)
   
def get_transfer_func(sys):
    """returns num,den"""
    return signal.ss2tf(sys.A,sys.B,sys.C,sys.D)
 
def compute_dynamic_pressure(vel):
    #dynamic pressure, rho
    rho = 1.204 #kg/m^2
    q = 0.5*vel**2*rho
   
    return q
 
       
def convert_knots_to_ms(knot_val):
    """convert knots to meters per second"""
    return knot_val*0.514444
 
def pretty_print(A, mode_type):
    print("Eigen Values for " + mode_type, A)
    print("\n")
   
   
def get_aircraft_info():
    folder_name = "aircraft_information"
    path = os.getcwd() + "/"+folder_name
   
    #get all csvs and compile to dataframe
    all_csv_files = util_functions.get_all_csv_files(path)
    df_list = util_functions.return_csv_list(all_csv_files)
   
    return df_list
 
def get_navion():
    df_list = get_aircraft_info()
    navion_lon_158 = df_list[1].iloc[0]
    navion_lat_properties = df_list[0].iloc[0]
    navion_phys_properties = df_list[2].iloc[0]
    navion = Aircraft(phys_df=navion_phys_properties, lon_df=navion_lon_158, lat_df=navion_lat_properties)
   
    return navion
 
def get_f104():
    ##Define F104
    df_list = get_aircraft_info()
    F104_lon = df_list[1].iloc[1]
    F104_lat_properties = df_list[0].iloc[1]
    F104_phys_properties = df_list[2].iloc[1]
    F104 = Aircraft(phys_df=F104_phys_properties, lon_df=F104_lon, lat_df=F104_lat_properties)
   
    return F104
 
if __name__=='__main__':
   
    """
    NOMENCLATURE
    L = ROLL MOMENT
    L_p is Roll damping
    C_lp = non dimensional roll damping stability derivative
    L_da, describes how much roll moment is generated by deflecting the ailerons
    C_lda is the aileron effectiveness
 
    L_p = (S*b^2*C_lp)/(2*I_x) * (dynamic_pressure/ Velocity)    
    L_da = (q*S*b*C_lda)/ Ix  
    C_lda = C_lda_right + C_lda_left # this is for the left and right aircraft
    """
 
    folder_name = "aircraft_information"
    path = os.getcwd() + "/"+folder_name
   
    #get all csvs and compile to dataframe
    all_csv_files = util_functions.get_all_csv_files(path)
    df_list = util_functions.return_csv_list(all_csv_files)
   
#%% INPUT PARAMETERS
    knot_min = 50
    knot_max = 200
    speed_min = convert_knots_to_ms(knot_min)
    speed_max = convert_knots_to_ms(knot_max)
    #vel_array = np.arange(speed_min, speed_max+1.0, 10)
    vel = 50
    q = 0.5*vel**2*1.204 * 0.5
    q_array = vel**2*1.204 * 0.5
    max_delta_a = math.radians(15)
    max_delta_r = 0.39
   
#%% Define aircraft    
    """refactor to automatically map to the aircraft"""
    plt.close('all')
    ##Define navion properties class
    navion_lon_158 = df_list[1].iloc[0]
    navion_lat_properties = df_list[0].iloc[0]
    navion_phys_properties = df_list[2].iloc[0]
    navion = Aircraft(phys_df=navion_phys_properties, lon_df=navion_lon_158, lat_df=navion_lat_properties)
   
    navion_dynamics = AircraftStateSpace(navion)
   
    time_list = []
    response_list = []
    zero_list = []
    pole_list = []
    A_list = []
    # for q,vel in zip(q_array, vel_array):
    A = navion_dynamics.compute_roll_mode(q,vel)
    #eig = np.linalg.eig(A)
    #pretty_print(A, "roll mode")
   
    A = navion_dynamics.compute_short_period_mode(max_delta_a, q, vel)
    eig = np.linalg.eig(A)
    pretty_print(eig[0], "short period mode")
   
    A = navion_dynamics.compute_dutch_roll_mode(alpha=max_delta_a, q=q, velocity=vel, delta_r=max_delta_r)
    eig = np.linalg.eig(A)
    pretty_print(eig[0], "dutch roll mode")
   
   
    A = navion_dynamics.compute_long_dynamics(alpha=max_delta_a, q=q,velocity=vel)
    eig = np.linalg.eig(A)
    pretty_print(eig[0], "long dynamics")
   
   
    A = navion_dynamics.compute_lat_dynamics(alpha=max_delta_a, q=q, velocity=vel)
    eig = np.linalg.eig(A)
    pretty_print(eig[0], "lat dynamics")
   
    #time_list.append(t)
    #response_list.append(r)
 
    eigen_values = []
    for a in pole_list:
        eig = np.linalg.eig(a)
        eigen_values.append(eig)
 
    # plot = plot_hw.Plotter()
   
    # response_list = [abs(r) for r in response_list]      
 
    # plot.plot_system_response(x_list=time_list, y_list=response_list,
    #                             line_labels=[str(vel) for vel in vel_array],
    #                             title_name='Short Period Mode', x_label='Hello', y_label='World', state_num=2)
   
    # plot.plot_poles_zeros(gain_list=pole_list[7:8], line_labels=[str(vel) for vel in vel_array],
    #                       title_name='Lateral Dynamics, theta_dot')
 
    # ##Define F104
    # F104_lon = df_list[1].iloc[1]
    # F104_lat_properties = df_list[0].iloc[1]
    # F104_phys_properties = df_list[2].iloc[1]
    # F104 = Aircraft(phys_df=F104_phys_properties, lon_df=F104_lon, lat_df=F104_lat_properties)
   
   
    #%%
    #F104.compute_Z_alpha(5, val=2, test=2)
   