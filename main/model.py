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
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Modelling a feed-forward loop network motif", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)
st.title("Modelling a feed-forward loop network motif | M. Hirsch, P. Kittler")

st.sidebar.write("Show graphs:")
sb_showgraph_colx, sb_showgraph_coly, sb_showgraph_colz = st.sidebar.columns(3)
with sb_showgraph_colx:
    show_x_active = st.checkbox('X*', value=False)
with sb_showgraph_coly:
    show_y = st.checkbox('Y', value=False)
with sb_showgraph_colz:
    show_z = st.checkbox('Z', value=True)

# parameters
x_active = 0
y_0 = 0
z_0 = 0

st.sidebar.subheader("Parameters")

k_xy = st.sidebar.slider("K_xy", min_value=0.01, max_value=1., value=1., step=0.01)   # activation / repression coefficient XY
k_xz = st.sidebar.slider("K_xz", min_value=0.01, max_value=1., value=1., step=0.01)   # activation / repression coefficient XZ
k_yz = st.sidebar.slider("K_yz", min_value=0.01, max_value=1., value=1., step=0.01)   # activation / repression coefficient YZ

a_y = st.sidebar.slider("a_y", min_value=0.01, max_value=1., value=1., step=0.01)
a_z = st.sidebar.slider("a_z", min_value=0.01, max_value=1., value=1., step=0.01)

b_y = st.sidebar.slider("B_x", min_value=0., max_value=10., value=0., step=0.1)     # basal concentration of Y
b_z = st.sidebar.slider("B_y", min_value=0., max_value=10., value=0., step=0.1)     # basal concentration of Z

beta_y = st.sidebar.slider("beta_y", min_value=0., max_value=1., value=1., step=0.01)
beta_z = st.sidebar.slider("beta_z", min_value=0., max_value=1., value=1., step=0.01)

h = st.sidebar.slider("H", min_value=0, max_value=3, value=2)                       # exponent of Hill function

t_interval = st.sidebar.slider("S_x active time interval", 1.0, 100.0, (10., 15.))         # time endpoint
steps = st.sidebar.slider("steps", min_value=1, max_value=300, value=200)        # number of steps/points in calculation

# function for control the visibility of graphs
def Plot_Control():
    output = []
    if(show_x_active == True):
        output.append("c[X*]")

    if(show_y == True):
        output.append("c[Y]")

    if(show_z == True):
        output.append("c[Z]")

    if(len(output) == 1):
        reduced_output = output[0]
        print("length:", len(output))
        return reduced_output
    else:
        print("length:", len(output))
        return output

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
    x = x_active             # switch for turn X on/off
    return b_y + beta_y * Regulation(x, k_xy, h, isActivator=flag_activator) - a_y * y

# function for time derivative of Z
def ODE_Z (t, z, y,  x_active, k_xz, k_yz, a_z, b_z, beta_z, h, s_x, flag_activator):
    x = s_x * x_active              # switch for turn X on/off
    return b_z + beta_z * Competitive_Regulation(x, y, k_xz, k_yz, h, isActivator=flag_activator) - a_z * z

# function for model to merge ODE_Y and ODE_Z to a ODE system
def Model (t, init_vars, k_xy, k_xz, k_yz, a_y, b_y, beta_y, a_z, b_z, beta_z, h, s_x, flag_activator):
    x_active, y, z = init_vars
    if(s_x == 1):
        x_active = 1
    else:
        x_active = 0

    dxdt = 0                        # if X is constant, then change rate is 0
    dydt = ODE_Y(t, y, x_active, k_xy, a_y, b_y, beta_y, h, s_x, flag_activator)
    dzdt = ODE_Z(t, z, y, x_active, k_xz, k_yz, a_z, b_z, beta_z, h, s_x, flag_activator)

    return [dxdt, dydt, dzdt]

def CalculateType(s_x, s_y):
    initial_values = [s_x, s_y, 0]
    # preparation for plotting
    t_span = t_interval[1] - t_interval[0]
    x_active_values = []
    y_values = []
    z_values = []
    s_x_values = []
    t_values = np.linspace(0, 100, steps)

    for t_step in range(steps):
        t = (100 / steps) * t_step

        if (t > t_interval[0] and t < t_interval[1]):
            x_active = s_x
            s_x_values.append(s_x)  # write s_x in list
        else:
            x_active = 0
            s_x_values.append(0)  # write s_x in list

        model_solver = solve_ivp(Model, (t, t + t_values[-1] / steps), initial_values, args=(k_xy, k_xz, k_yz, a_y, b_y, beta_y, a_z, b_z, beta_z, h, s_x_values[-1], True), t_eval=[t + t_span / steps])

        # update initial value (conditions)
        initial_values = [model_solver.y[0][-1], model_solver.y[1][-1], model_solver.y[2][-1]]

        x_active_values.append(model_solver.y[0][0])  # save only first value from last step
        y_values.append(model_solver.y[1][0])  # save only first value from last step
        z_values.append(model_solver.y[2][0])  # save only first value from last step

    # pack the results in 1 output object
    output_object = [t_values, x_active_values, y_values, z_values, s_x_values]

    return output_object

# COHERENT TYPE 1

CT1 = CalculateType(s_x = 1, s_y = 1)

dframe_coherent_type_1 = pd.DataFrame({
    'time': CT1[0],
    'c[X*]': CT1[1],
    'c[Y]': CT1[2],
    'c[Z]': CT1[3]
})

dframe_coherent_type_1_sx = pd.DataFrame({
    'time': CT1[0],
    'S_x': CT1[4]
})

container_coherent = st.container()
col_coherent_type_1, col_coherent_type_2, col_coherent_type_3, col_coherent_type_4 = container_coherent.columns(4)

with col_coherent_type_1:
    st.subheader("Coherent Type 1")
    tab_plot, tab_data = st.tabs(["Plot", "Data"])

with tab_plot:
    st.line_chart(dframe_coherent_type_1_sx, x="time", y="S_x")
    st.line_chart(dframe_coherent_type_1, x="time", y=Plot_Control())
with tab_data:
    dframe_coherent_type_1

with col_coherent_type_2:
    st.subheader("Coherent Type 2")
    tab_plot, tab_data = st.tabs(["Plot", "Data"])

with tab_plot:
    st.line_chart(df, x="time", y=Plot_Control())
with tab_data:
    df

with col_coherent_type_3:
    st.subheader("Coherent Type 3")
    tab_plot, tab_data = st.tabs(["Plot", "Data"])

with tab_plot:
    st.line_chart(df, x="time (s)", y=Plot_Control())
with tab_data:
    df

with col_coherent_type_4:
    st.subheader("Coherent Type 4")
    tab_plot, tab_data = st.tabs(["Plot", "Data"])

with tab_plot:
    st.line_chart(df, x="time (s)", y=Plot_Control())
with tab_data:
    df

container_incoherent = st.container()
col_incoherent_type_1, col_incoherent_type_2, col_incoherent_type_3, col_incoherent_type_4 = container_incoherent.columns(4)

with col_incoherent_type_1:
    st.subheader("Incoherent Type 1")
    tab_plot, tab_data = st.tabs(["Plot", "Data"])

with tab_plot:
    st.line_chart(df, x="time (s)", y=Plot_Control())
with tab_data:
    df

with col_incoherent_type_2:
    st.subheader("Incoherent Type 2")
    tab_plot, tab_data = st.tabs(["Plot", "Data"])

with tab_plot:
    st.line_chart(df, x="time (s)", y=Plot_Control())
with tab_data:
    df

with col_incoherent_type_3:
    st.subheader("Incoherent Type 3")
    tab_plot, tab_data = st.tabs(["Plot", "Data"])

with tab_plot:
    st.line_chart(df, x="time (s)", y=Plot_Control())
with tab_data:
    df

with col_incoherent_type_4:
    st.subheader("Incoherent Type 4")
    tab_plot, tab_data = st.tabs(["Plot", "Data"])

with tab_plot:
    st.line_chart(df, x="time (s)", y=Plot_Control())
with tab_data:
    df