import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import streamlit as st
import pandas as pd

# COHERENT TYPE 1
print("--- --- COHERENT TYPE 1 --- ---")
def CT_1_AND_System(t, init_vars, k_xz, k_xy, k_yz, a_y, a_z, b_y, b_z, beta_y, beta_z, h, s_x, s_y, current_x, current_y, current_z):
    x, y, z = init_vars
    x = x * s_x
    y = x * s_y
    z = current_z

    dxdt = 0
    dydt = b_y + beta_y * (((x / k_xy)**h) / (1 + (x / k_xy) ** h)) - a_y * y
    dzdt = b_z + beta_z * (((x / k_xz)**h) / (1 + (x / k_xz) ** h)) * (((y / k_yz)**h) / (1 + (y / k_yz) ** h)) - a_z * z

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
        t = (t_span / steps) * t_step

        if (t <= t_span):
            s_x_values.append(s_x)  # write s_x in list
            s_y_values.append(s_y)  # write s_y in list
        else:
            s_x_values.append(0)    # write s_x in list
            s_y_values.append(0)    # write s_y in list

        if t_step < 1:
            model_solver = solve_ivp(CT_1_AND_System, (t, t + t_span / steps), initial_values, args=(k_xz, k_xy, k_yz, a_y, a_z, b_y, b_z, beta_y, beta_z, h, s_x_values[-1], s_y_values[-1], X0, Y0, Z0), t_eval=[t + t_span / steps])
        else:
            model_solver = solve_ivp(CT_1_AND_System, (t, t + t_span / steps), initial_values, args=(k_xz, k_xy, k_yz, a_y, a_z, b_y, b_z, beta_y, beta_z, h, s_x_values[-1], s_y_values[-1], x_values[-1], y_values[-1], z_values[-1]), t_eval=[t + t_span / steps])

        initial_values = [model_solver.y[0][-1], model_solver.y[1][-1], model_solver.y[2][-1]]

        x_values.append(model_solver.y[0][0])  # save only first value from last step
        y_values.append(model_solver.y[1][0])  # save only first value from last step
        z_values.append(model_solver.y[2][0])  # save only first value from last step

    return [t_values, x_values, y_values, z_values, s_x_values, s_y_values]

ct_1_and_data_set_1 = CT_1_AND(0., 0., 0.1, 0.1, 0.5, 10., 1., 1.)
ct_1_and_data_set_2 = CT_1_AND(0., 0., 0.1, 0.1, 5, 10., 1., 1.)

fig_ct1_and, ax_ct1_and = plt.subplots(1,1)
ax_ct1_and.plot(ct_1_and_data_set_1[3], label="k_yz = 0.5")
ax_ct1_and.plot(ct_1_and_data_set_2[3], label="k_yz = 5")
ax_ct1_and.plot(ct_1_and_data_set_1[4], label="s_x")
ax_ct1_and.plot(ct_1_and_data_set_1[5], label="s_y")
ax_ct1_and.legend()
fig_ct1_and.show()


# COHERENT TYPE 4
print("--- --- COHERENT TYPE 4 --- ---")
def CT_4_AND_System(t, init_vars, k_xz, k_xy, k_yz, a_y, a_z, b_y, b_z, beta_y, beta_z, h, s_x, s_y, current_x, current_y, current_z):
    x, y, z = init_vars
    x = x * s_x
    y = x * s_y
    z = current_z

    dxdt = 0
    dydt = b_y + beta_y * (1 / (1 + (x / k_xy) ** h)) - a_y * y
    dzdt = b_z + beta_z * (((x / k_xz)**h) / (1 + (x / k_xz) ** h)) * (1 / (1 + (y / k_yz) ** h)) - a_z * z

    return [dxdt, dydt, dzdt]

def CT_4_AND(b_y, b_z, k_xz, k_xy, k_yz, t_span, s_x, s_y):
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

        if t_step < 1:
            model_solver = solve_ivp(CT_4_AND_System, (t, t + t_span / steps), initial_values, args=(k_xz, k_xy, k_yz, a_y, a_z, b_y, b_z, beta_y, beta_z, h, s_x_values[-1], s_y_values[-1], X0, Y0, Z0), t_eval=[t + t_span / steps])
        else:
            model_solver = solve_ivp(CT_4_AND_System, (t, t + t_span / steps), initial_values, args=(k_xz, k_xy, k_yz, a_y, a_z, b_y, b_z, beta_y, beta_z, h, s_x_values[-1], s_y_values[-1], x_values[-1], y_values[-1], z_values[-1]), t_eval=[t + t_span / steps])

        initial_values = [model_solver.y[0][-1], model_solver.y[1][-1], model_solver.y[2][-1]]

        x_values.append(model_solver.y[0][0])  # save only first value from last step
        y_values.append(model_solver.y[1][0])  # save only first value from last step
        z_values.append(model_solver.y[2][0])  # save only first value from last step

    return [t_values, x_values, y_values, z_values, s_x_values, s_y_values]

ct_4_and_data_set_1 = CT_4_AND(0., 0., 0.1, 0.1, 0.6, 10., 1., 1.)
ct_4_and_data_set_2 = CT_4_AND(0., 0., 0.1, 0.1, 0.3, 10., 1., 1.)

fig_ct4_and, ax_ct4_and = plt.subplots(1,1)
ax_ct4_and.plot(ct_4_and_data_set_1[3], label="k_yz = 0.6")
ax_ct4_and.plot(ct_4_and_data_set_2[3], label="k_yz = 0.3")
ax_ct4_and.plot(ct_4_and_data_set_1[4], label="s_x")
ax_ct4_and.plot(ct_4_and_data_set_1[5], label="s_y")
ax_ct4_and.legend()
fig_ct4_and.show()

# INCOHERENT TYPE 1
print("--- --- INCOHERENT TYPE 1 --- ---")
def IT_1_AND_System(t, init_vars, k_xz, k_xy, k_yz, a_y, a_z, b_y, b_z, beta_y, beta_z, h, s_x, s_y, current_x, current_y, current_z):
    x, y, z = init_vars
    x = x * s_x
    y = x * s_y
    z = current_z

    dxdt = 0
    dydt = b_y + beta_y * ((x / k_xy) ** h / (1 + (x / k_xy) ** h)) - a_y * y
    dzdt = b_z + beta_z * (((x / k_xz)**h) / (1 + (x / k_xz) ** h)) * (1 / (1 + (y / k_yz) ** h)) - a_z * z

    return [dxdt, dydt, dzdt]

def IT_1_AND(b_y, b_z, k_xz, k_xy, k_yz, t_span, s_x, s_y):
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

        if t_step < 1:
            model_solver = solve_ivp(IT_1_AND_System, (t, t + t_span / steps), initial_values, args=(k_xz, k_xy, k_yz, a_y, a_z, b_y, b_z, beta_y, beta_z, h, s_x_values[-1], s_y_values[-1], X0, Y0, Z0), t_eval=[t + t_span / steps])
        else:
            model_solver = solve_ivp(IT_1_AND_System, (t, t + t_span / steps), initial_values, args=(k_xz, k_xy, k_yz, a_y, a_z, b_y, b_z, beta_y, beta_z, h, s_x_values[-1], s_y_values[-1], x_values[-1], y_values[-1], z_values[-1]), t_eval=[t + t_span / steps])

        initial_values = [model_solver.y[0][-1], model_solver.y[1][-1], model_solver.y[2][-1]]

        x_values.append(model_solver.y[0][0])  # save only first value from last step
        y_values.append(model_solver.y[1][0])  # save only first value from last step
        z_values.append(model_solver.y[2][0])  # save only first value from last step

    return [t_values, x_values, y_values, z_values, s_x_values, s_y_values]

it_1_and_data_set_1 = IT_1_AND(0., 0., 0.1, 0.1, 0.01, 10., 1., 1.)
it_1_and_data_set_2 = IT_1_AND(0., 0., 0.1, 0.1, 0.1, 10., 1., 1.)
it_1_and_data_set_3 = IT_1_AND(0., 0., 0.1, 0.1, 0.3, 10., 1., 1.)

fig_it1_and, ax_it1_and = plt.subplots(1,1)
ax_it1_and.plot(it_1_and_data_set_1[3], label="k_yz = 0.01")
ax_it1_and.plot(it_1_and_data_set_2[3], label="k_yz = 0.1")
ax_it1_and.plot(it_1_and_data_set_3[3], label="k_yz = 0.3")
ax_it1_and.plot(it_1_and_data_set_1[4], label="s_x")
ax_it1_and.plot(it_1_and_data_set_1[5], label="s_y")
ax_it1_and.legend()
fig_it1_and.show()