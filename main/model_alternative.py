import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import streamlit as st
import pandas as pd

# COHERENT TYPE 1
print("--- --- COHERENT TYPE 1 --- ---")
def CT_1_AND_System(k_xz, k_xy, k_yz, a_y, a_z, b_y, b_z, beta_y, beta_z, h, s_x, s_y, current_x, current_y, current_z):
    x = current_x * s_x
    y = current_y * s_y
    z = current_z

    dxdt = 0
    dydt = b_y + beta_y * (((x / k_xy)**h) / (1 + (x / k_xy) ** h)) - a_y * y
    dzdt = b_z + beta_z * (((x / k_xz)**h) / (1 + (x / k_xz) ** h)) * (((y / k_yz)**h) / (1 + (y / float(k_yz)) ** h)) - a_z * z

    return [dxdt, dydt, dzdt]

def CT_1_AND(b_y, b_z, k_xz, k_xy, k_yz, t_span, s_x, s_y):
    X0 = 1
    Y0 = 0
    Z0 = 0

    h = 2

    a_y = 1
    a_z = 1

    beta_y = 1
    beta_z = 1

    steps = 100

    x_values = []
    y_values = []
    z_values = []

    t_values = np.linspace(0, 20, steps)

    s_x_values = []
    s_y_values = []

    initial_values = [X0, b_y, b_z]

    for t_step in range(steps):
        t = (20 / steps) * t_step

        if (t <= t_span):
            s_x_values.append(s_x)  # write s_x in list
            s_y_values.append(s_y)  # write s_y in list
        else:
            s_x_values.append(0)    # write s_x in list
            s_y_values.append(0)    # write s_y in list

        if t_step < 2:
            model_solver = solve_ivp(CT_1_AND_System, (t, t + t_span / steps), initial_values, args=(k_xz, k_xy, k_yz, a_y, a_z, b_y, b_z, beta_y, beta_z, h, s_x_values[-1], s_y_values[-1], X0, Y0), t_eval=[t + t_span / steps])
        else:
            model_solver = solve_ivp(CT_1_AND_System, (t, t + t_span / steps), initial_values, args=(k_xz, k_xy, k_yz, a_y, a_z, b_y, b_z, beta_y, beta_z, h, s_x_values[-1], s_y_values[-1], x_values[-1], y_values[-1], z_values[-1]), t_eval=[t + t_span / steps])

        initial_values = [model_solver.y[0][-1], model_solver.y[1][-1], model_solver.y[2][-1]]

        x_values.append(model_solver.y[0][0])  # save only first value from last step
        y_values.append(model_solver.y[1][0])  # save only first value from last step
        z_values.append(model_solver.y[2][0])  # save only first value from last step

    return [t_values, x_values, y_values, z_values, s_x_values, s_y_values]

ct_1_and_data_set_1 = CT_1_AND(0., 0., 0.1, 0.1, 0.5, 10., 1., 1.)