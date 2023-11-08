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
    show_x_active = st.checkbox('X*', value=True)
with sb_showgraph_coly:
    show_y = st.checkbox('Y', value=True)
with sb_showgraph_colz:
    show_z = st.checkbox('Z', value=True)

# parameters
st.sidebar.subheader("Parameters")
s_x = st.sidebar.slider("S_x", min_value=0, max_value=1, value=0, step=1)     # input for X
s_y = st.sidebar.slider("S_y", min_value=0, max_value=1, value=0, step=1)     # input for Y

x_active = st.sidebar.slider("X*", min_value=0., max_value=1., value=1., step=0.01)        # initial value for X
y_0 = st.sidebar.slider("Y0", min_value=0., max_value=1., value=0., step=0.01)             # initial value for Y
z_0 = st.sidebar.slider("Z0", min_value=0., max_value=1., value=0., step=0.01)             # initial value for Z

k_xy = st.sidebar.slider("K_xy", min_value=0., max_value=1., value=1., step=0.01)   # activation / repression coefficient XY
k_xz = st.sidebar.slider("K_xz", min_value=0., max_value=1., value=1., step=0.01)   # activation / repression coefficient XZ
k_yz = st.sidebar.slider("K_yz", min_value=0., max_value=1., value=1., step=0.01)   # activation / repression coefficient YZ

a_y = st.sidebar.slider("a_y", min_value=0., max_value=1., value=1., step=0.01)
a_z = st.sidebar.slider("a_z", min_value=0., max_value=1., value=1., step=0.01)

b_y = st.sidebar.slider("B_x", min_value=0., max_value=10., value=0., step=0.1)     # basal concentration of Y
b_z = st.sidebar.slider("B_y", min_value=0., max_value=10., value=0., step=0.1)     # basal concentration of Z

beta_y = st.sidebar.slider("beta_y", min_value=0., max_value=1., value=1., step=0.01)
beta_z = st.sidebar.slider("beta_z", min_value=0., max_value=1., value=1., step=0.01)

h = st.sidebar.slider("H", min_value=0, max_value=3, value=2)                       # exponent of Hill function

t_interval = st.sidebar.slider("time interval", 1.0, 100.0, (10., 15.))         # time endpoint
steps = st.sidebar.slider("steps", min_value=1, max_value=300, value=200)        # number of steps/points in calculation

initial_values = [x_active, y_0, z_0]

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

# preparation for plotting
t_span = t_interval[1] - t_interval[0]
x_active_values = []
y_values = []
z_values = []
s_x_values = []
t_values = np.linspace(0, 100, steps)

time_switch = [5., 10., 15., 25.]

# loop for calculation of results for each step
for t_step in range(steps):
    t = (100 / steps) * t_step

    current_s_x = s_x
    if(t > t_interval[0] and t < t_interval[1]):
        s_x_values.append(s_x)              # write s_x in list
        current_s_x = 1
    else:
        s_x_values.append(s_x - 1)
        current_s_x = 0

    model_solver = solve_ivp(Model, (t_interval[0], t_interval[1]), initial_values, args = (k_xy, k_xz, k_yz, a_y, b_y, beta_y, a_z, b_z, beta_z, h, current_s_x, True), t_eval = [t_interval[1]])
    # update initial value (conditions)
    initial_values = [model_solver.y[0][-1], model_solver.y[1][-1], model_solver.y[2][-1]]

    x_active_values.append(model_solver.y[0][0])    # save only first value from last step
    y_values.append(model_solver.y[1][0])           # save only first value from last step
    z_values.append(model_solver.y[2][0])           # save only first value from last step

df = pd.DataFrame({
    'time (s)': t_values,
    'c[X*]': x_active_values,
    'c[Y]': y_values,
    'c[Z]': z_values
})

dframe_coherent_type_1 = pd.DataFrame({
    'time (s)': t_values,
    'c[X*]': x_active_values,
    'c[Y]': y_values,
    'c[Z]': z_values
})

dframe_coherent_type_1_sx = pd.DataFrame({
    'time (s)': t_values,
    'S_x': s_x_values
})

container_coherent = st.container()
col_coherent_type_1, col_coherent_type_2, col_coherent_type_3, col_coherent_type_4 = container_coherent.columns(4)

with col_coherent_type_1:
    st.subheader("Coherent Type 1")
    tab_plot, tab_data = st.tabs(["Plot", "Data"])

with tab_plot:
    st.line_chart(dframe_coherent_type_1_sx, x="time (s)", y="S_x")
    st.line_chart(dframe_coherent_type_1, x="time (s)", y=["c[X*]", "c[Y]", "c[Z]"])
with tab_data:
    dframe_coherent_type_1

with col_coherent_type_2:
    st.subheader("Coherent Type 2")
    tab_plot, tab_data = st.tabs(["Plot", "Data"])

with tab_plot:
    st.line_chart(df, x="time (s)", y=["c[X*]", "c[Y]", "c[Z]"])
with tab_data:
    df

with col_coherent_type_3:
    st.subheader("Coherent Type 3")
    tab_plot, tab_data = st.tabs(["Plot", "Data"])

with tab_plot:
    st.line_chart(df, x="time (s)", y=["c[X*]", "c[Y]", "c[Z]"])
with tab_data:
    df

with col_coherent_type_4:
    st.subheader("Coherent Type 4")
    tab_plot, tab_data = st.tabs(["Plot", "Data"])

with tab_plot:
    st.line_chart(df, x="time (s)", y=["c[X*]", "c[Y]", "c[Z]"])
with tab_data:
    df

container_incoherent = st.container()
col_incoherent_type_1, col_incoherent_type_2, col_incoherent_type_3, col_incoherent_type_4 = container_incoherent.columns(4)

with col_incoherent_type_1:
    st.subheader("Incoherent Type 1")
    tab_plot, tab_data = st.tabs(["Plot", "Data"])

with tab_plot:
    st.line_chart(df, x="time (s)", y=["c[X*]", "c[Y]", "c[Z]"])
with tab_data:
    df

with col_incoherent_type_2:
    st.subheader("Incoherent Type 2")
    tab_plot, tab_data = st.tabs(["Plot", "Data"])

with tab_plot:
    st.line_chart(df, x="time (s)", y=["c[X*]", "c[Y]", "c[Z]"])
with tab_data:
    df

with col_incoherent_type_3:
    st.subheader("Incoherent Type 3")
    tab_plot, tab_data = st.tabs(["Plot", "Data"])

with tab_plot:
    st.line_chart(df, x="time (s)", y=["c[X*]", "c[Y]", "c[Z]"])
with tab_data:
    df

with col_incoherent_type_4:
    st.subheader("Incoherent Type 4")
    tab_plot, tab_data = st.tabs(["Plot", "Data"])

with tab_plot:
    st.line_chart(df, x="time (s)", y=["c[X*]", "c[Y]", "c[Z]"])
with tab_data:
    df

'''
# plot the results in one plot
plt.figure(figsize=(12, 8))

# 1st plot: turn s_x on and off
plt.subplot(2, 1, 1)
plt.plot(t_values, s_x_values, label='S_x')
plt.xlabel('time')
plt.ylabel('S_x')
plt.title('turn S_x on and off')
plt.grid(True)

# 2nd plot: show concentration of x_active, y and z
plt.subplot(2, 1, 2)
plt.plot(t_values, x_active_values, label='X*')
plt.plot(t_values, y_values, label='Y')
plt.plot(t_values, z_values, label='Z')
plt.xlabel('time')
plt.ylabel('concentration')
plt.legend()
plt.title('Results of ODEs')
plt.grid(True)

plt.tight_layout()                  # arrange position of subplots to prevent overlaps

plt.show()
'''