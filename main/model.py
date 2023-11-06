'''
file        model.py
description feed-forward-loop model based upon publication by U. Alon
date        2023-11-06
version     0.1
authors     Max Hirsch, Philipp Kittler
license     usage, modification and redistribution only for private, educational and scientific purposes, redistribution only without third party dependencies
'''

# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# parameters
s_x = 0             # input for X
s_y = 0             # input for Y

k_xy = 1            # activation / repression coefficient XY
k_xz = 1            # activation / repression coefficient XZ
k_yz = 1            # activation / repression coefficient YZ

b_y = 0             # basal concentration of Y
b_z = 0             # basal concentration of Z

a_y = 1
a_z = 1

beta_y = 1
beta_z = 1

h = 2               # exponent of Hill function

x_active = 1        # initial value for X
y_0 = 0             # initial value for Y
z_0 = 0             # initial value for Z

initial_values = [x_active, y_0, z_0]

t_end = 20          # time endpoint

steps = 200         # number of steps/points in calculation

# function for regulation, named f in paper
def Regulation (u, k, h, isActivator = True):
    if isActivator == True:
        return (((u / k)**h) / (1 + (u / k) ** h))
    else:
        return (1 / (1 + (u / k) ** h))

# function for competitive regulation, named f_c in paper
def Competitive_Regulation (u, v, k_u, k_v, h, isActivator = True):
    if isActivator == True:
        return (((u / k_u) ** h) / (1 + (u / k_u) ** h + (v / k_v) ** h))
    else:
        return (1/ (1 + (u / k_u) ** h + (v / k_v) ** h))

# function for time derivative of Y
def ODE_Y (t, y, x_active, k_xy, a_y, b_y, beta_y, h, s_x, flag_activator):
    x = s_x * x_active              # switch for turn X on/off
    return b_y + beta_y * Regulation(x, k_xy, h, isActivator=flag_activator) - a_y * y

# function for time derivative of Z
def ODE_Z (t, z, y,  x_active, k_xz, k_yz, a_z, b_z, beta_z, h, s_x, flag_activator):
    x = s_x * x_active              # switch for turn X on/off
    return b_z + beta_z * Competitive_Regulation(x, y, k_xz, k_yz, h, isActivator=flag_activator) - a_z * z

# function for model to merge ODE_Y and ODE_Z to a ODE system
def Model (t, init_vars, k_xy, k_xz, k_yz, a_y, b_y, beta_y, a_z, b_z, beta_z, h, s_x, flag_activator):
    x_active, y, z = init_vars

    dxdt = 0                        # if X is constant, then change rate is 0
    dydt = ODE_Y(t, y, x_active, k_xy, a_y, b_y, beta_y, h, s_x, flag_activator)
    dzdt = ODE_Z(t, z, y, x_active, k_xz, k_yz, a_z, b_z, beta_z, h, s_x, flag_activator)

    return [dxdt, dydt, dzdt]

