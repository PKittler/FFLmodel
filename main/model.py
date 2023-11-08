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
    show_y = st.checkbox('Y', value=True)
with sb_showgraph_colz:
    show_z = st.checkbox('Z', value=True)

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False
    st.session_state.horizontal = True

# parameters
x_active = 0
y_0 = 0
z_0 = 0

st.sidebar.subheader("Parameters")

c_logicmode = st.sidebar.radio("Coherent Logic Mode", ["AND", "OR"], horizontal=st.session_state.horizontal)

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

t_interval = st.sidebar.slider("Inducer active time interval", 1.0, 100.0, (10., 15.))         # time endpoint
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
def Regulation (u, k, h, isActivator):
    if isActivator == True:
        return (((u / k)**h) / (1 + (u / k) ** h))
    else:
        return (1 / (1 + (u / k) ** h))

# function for competitive regulation, named f_c in paper
def Competitive_Regulation (u, v, k_u, k_v, h, isUActivator, isVActivator):
    if isUActivator == True:
        return (((u / k_u) ** h) / (1 + (u / k_u) ** h + (v / k_v) ** h))
    else:
        return (1/ (1 + (u / k_u) ** h + (v / k_v) ** h))

def Gate(x, y, k_xz, k_yz, h, isUActivator, isVActivator):
    if(c_logicmode == "AND"):
        return Regulation(x, k_xz, h, isUActivator) * Regulation(y, k_yz, h, isVActivator)
    elif(c_logicmode == "OR"):
        return Competitive_Regulation(x, y, k_xz, k_yz, h, isUActivator, isVActivator) + Competitive_Regulation(y, x, k_yz, k_xz, h, isUActivator, isVActivator)

# function for time derivative of Y
def ODE_Y (t, y, x_active, k_xy, a_y, b_y, beta_y, h, s_x, flag_activator_y):
    x = x_active             # switch for turn X on/off
    return b_y + beta_y * Regulation(x, k_xy, h, isActivator=flag_activator_y) - a_y * y

# function for time derivative of Z
def ODE_Z (t, z, y,  x_active, k_xz, k_yz, a_z, b_z, beta_z, h, s_x, flag_activator_x, flag_activator_y):
    x = s_x * x_active              # switch for turn X on/off
    return b_z + beta_z * Gate(x, y, k_xz, k_yz, h, isUActivator=flag_activator_x, isVActivator=flag_activator_y) - a_z * z

# function for model to merge ODE_Y and ODE_Z to a ODE system
def Model (t, init_vars, k_xy, k_xz, k_yz, a_y, b_y, beta_y, a_z, b_z, beta_z, h, s_x, s_y, flag_activator_x, flag_activator_y):
    x_active, y, z = init_vars
    x_active = s_x
    y = s_y

    dxdt = 0                        # if X is constant, then change rate is 0
    dydt = ODE_Y(t, y, x_active, k_xy, a_y, b_y, beta_y, h, s_x, flag_activator_y)
    dzdt = ODE_Z(t, z, y, x_active, k_xz, k_yz, a_z, b_z, beta_z, h, s_x, flag_activator_x, flag_activator_y)

    return [dxdt, dydt, dzdt]

def CalculateType(s_x, s_y, t_start, t_end, flag_activator_x, flag_activator_y):
    initial_values = [s_x, s_y, 0]

    t_span = t_end - t_start

    x_active_values = []
    y_values = []
    z_values = []

    s_x_values = []
    s_y_values = []

    t_values = np.linspace(0, 100, steps)

    for t_step in range(steps):
        t = (100 / steps) * t_step

        if (t > t_start and t < t_end):
            s_x_values.append(s_x)  # write s_x in list
            s_y_values.append(s_y)  # write s_y in list
        else:
            s_x_values.append(0)  # write s_x in list
            s_y_values.append(0)  # write s_y in list

        model_solver = solve_ivp(Model, (t, t + t_values[-1] / steps), initial_values, args=(k_xy, k_xz, k_yz, a_y, b_y, beta_y, a_z, b_z, beta_z, h, s_x_values[-1], s_y_values[-1], flag_activator_x, flag_activator_y), t_eval=[t + t_span / steps])

        # update initial value (conditions)
        initial_values = [model_solver.y[0][-1], model_solver.y[1][-1], model_solver.y[2][-1]]

        x_active_values.append(model_solver.y[0][0])  # save only first value from last step
        y_values.append(model_solver.y[1][0])  # save only first value from last step
        z_values.append(model_solver.y[2][0])  # save only first value from last step

    # pack the results in 1 output object
    output_object = [t_values, x_active_values, y_values, z_values, s_x_values, s_y_values]

    return output_object

# COHERENT TYPE 1

CT1 = CalculateType(s_x = 1, s_y = 1, t_start = t_interval[0], t_end = t_interval[1], flag_activator_x=True, flag_activator_y=True)

dframe_coherent_type_1 = pd.DataFrame({
    'time': CT1[0],
    'c[X*]': CT1[1],
    'c[Y]': CT1[2],
    'c[Z]': CT1[3]
})

dframe_coherent_type_1_sx = pd.DataFrame({
    'time': CT1[0],
    'S_x': CT1[4],
    'S_y': CT1[5]
})

# COHERENT TYPE 2

CT2 = CalculateType(s_x = 1, s_y = 1, t_start = t_interval[0], t_end = t_interval[1], flag_activator_x=True, flag_activator_y=False)

dframe_coherent_type_2 = pd.DataFrame({
    'time': CT2[0],
    'c[X*]': CT2[1],
    'c[Y]': CT2[2],
    'c[Z]': CT2[3]
})

dframe_coherent_type_2_sx = pd.DataFrame({
    'time': CT2[0],
    'S_x': CT2[4],
    'S_y': CT2[5]
})

# COHERENT TYPE 3

CT3 = CalculateType(s_x = 1, s_y = 1, t_start = t_interval[0], t_end = t_interval[1], flag_activator_x=False, flag_activator_y=True)

dframe_coherent_type_3 = pd.DataFrame({
    'time': CT3[0],
    'c[X*]': CT3[1],
    'c[Y]': CT3[2],
    'c[Z]': CT3[3]
})

dframe_coherent_type_3_sx = pd.DataFrame({
    'time': CT3[0],
    'S_x': CT3[4],
    'S_y': CT3[5]
})

# COHERENT TYPE 4

CT4 = CalculateType(s_x = 1, s_y = 1, t_start = t_interval[0], t_end = t_interval[1], flag_activator_x=False, flag_activator_y=False)

dframe_coherent_type_4 = pd.DataFrame({
    'time': CT4[0],
    'c[X*]': CT4[1],
    'c[Y]': CT4[2],
    'c[Z]': CT4[3]
})

dframe_coherent_type_4_sx = pd.DataFrame({
    'time': CT4[0],
    'S_x': CT4[4],
    'S_y': CT4[5]
})

container_coherent = st.container()
col_coherent_type_1, col_coherent_type_2, col_coherent_type_3, col_coherent_type_4 = container_coherent.columns(4)

with col_coherent_type_1:
    st.subheader("Coherent Type 1")
    tab_plot, tab_data = st.tabs(["Plot", "Data"])

with tab_plot:
    st.line_chart(dframe_coherent_type_1_sx, x="time", y=["S_x", "S_y"])
    st.line_chart(dframe_coherent_type_1, x="time", y=Plot_Control())
with tab_data:
    dframe_coherent_type_1


with col_coherent_type_2:
    st.subheader("Coherent Type 2")
    tab_plot, tab_data = st.tabs(["Plot", "Data"])

with tab_plot:
    st.line_chart(dframe_coherent_type_2_sx, x="time", y=["S_x", "S_y"])
    st.line_chart(dframe_coherent_type_2, x="time", y=Plot_Control())
with tab_data:
    dframe_coherent_type_2


with col_coherent_type_3:
    st.subheader("Coherent Type 3")
    tab_plot, tab_data = st.tabs(["Plot", "Data"])

with tab_plot:
    st.line_chart(dframe_coherent_type_3_sx, x="time", y=["S_x", "S_y"])
    st.line_chart(dframe_coherent_type_3, x="time", y=Plot_Control())
with tab_data:
    dframe_coherent_type_3


with col_coherent_type_4:
    st.subheader("Coherent Type 4")
    tab_plot, tab_data = st.tabs(["Plot", "Data"])

with tab_plot:
    st.line_chart(dframe_coherent_type_4_sx, x="time", y=["S_x", "S_y"])
    st.line_chart(dframe_coherent_type_4, x="time", y=Plot_Control())
with tab_data:
    dframe_coherent_type_4


# INCOHERENT TYPE 1

IT1 = CalculateType(s_x = 1, s_y = 1, t_start = t_interval[0], t_end = t_interval[1], flag_activator_x=True, flag_activator_y=True)

dframe_incoherent_type_1 = pd.DataFrame({
    'time': IT1[0],
    'c[X*]': IT1[1],
    'c[Y]': IT1[2],
    'c[Z]': IT1[3]
})

dframe_incoherent_type_1_sx = pd.DataFrame({
    'time': IT1[0],
    'S_x': IT1[4],
    'S_y': IT1[5]
})

# INCOHERENT TYPE 2

IT2 = CalculateType(s_x = 1, s_y = 1, t_start = t_interval[0], t_end = t_interval[1], flag_activator_x=True, flag_activator_y=False)

dframe_incoherent_type_2 = pd.DataFrame({
    'time': IT2[0],
    'c[X*]': IT2[1],
    'c[Y]': IT2[2],
    'c[Z]': IT2[3]
})

dframe_incoherent_type_2_sx = pd.DataFrame({
    'time': IT2[0],
    'S_x': IT2[4],
    'S_y': IT2[5]
})

# INCOHERENT TYPE 3

IT3 = CalculateType(s_x = 1, s_y = 1, t_start = t_interval[0], t_end = t_interval[1], flag_activator_x=False, flag_activator_y=True)

dframe_incoherent_type_3 = pd.DataFrame({
    'time': IT3[0],
    'c[X*]': IT3[1],
    'c[Y]': IT3[2],
    'c[Z]': IT3[3]
})

dframe_incoherent_type_3_sx = pd.DataFrame({
    'time': IT3[0],
    'S_x': IT3[4],
    'S_y': IT3[5]
})

# INCOHERENT TYPE 4

IT4 = CalculateType(s_x = 1, s_y = 1, t_start = t_interval[0], t_end = t_interval[1], flag_activator_x=False, flag_activator_y=False)

dframe_incoherent_type_4 = pd.DataFrame({
    'time': IT4[0],
    'c[X*]': IT4[1],
    'c[Y]': IT4[2],
    'c[Z]': IT4[3]
})

dframe_incoherent_type_4_sx = pd.DataFrame({
    'time': IT4[0],
    'S_x': IT4[4],
    'S_y': IT4[5]
})



container_incoherent = st.container()
col_incoherent_type_1, col_incoherent_type_2, col_incoherent_type_3, col_incoherent_type_4 = container_incoherent.columns(4)

with col_incoherent_type_1:
    st.subheader("Incoherent Type 1")
    tab_plot, tab_data = st.tabs(["Plot", "Data"])

with tab_plot:
    st.line_chart(dframe_incoherent_type_1_sx, x="time", y=["S_x", "S_y"])
    st.line_chart(dframe_incoherent_type_1, x="time", y=Plot_Control())
with tab_data:
    dframe_incoherent_type_1


with col_incoherent_type_2:
    st.subheader("Incoherent Type 2")
    tab_plot, tab_data = st.tabs(["Plot", "Data"])

with tab_plot:
    st.line_chart(dframe_incoherent_type_2_sx, x="time", y=["S_x", "S_y"])
    st.line_chart(dframe_incoherent_type_2, x="time", y=Plot_Control())
with tab_data:
    dframe_incoherent_type_2


with col_incoherent_type_3:
    st.subheader("Incoherent Type 3")
    tab_plot, tab_data = st.tabs(["Plot", "Data"])

with tab_plot:
    st.line_chart(dframe_incoherent_type_3_sx, x="time", y=["S_x", "S_y"])
    st.line_chart(dframe_incoherent_type_3, x="time", y=Plot_Control())
with tab_data:
    dframe_incoherent_type_3


with col_incoherent_type_4:
    st.subheader("Incoherent Type 4")
    tab_plot, tab_data = st.tabs(["Plot", "Data"])

with tab_plot:
    st.line_chart(dframe_incoherent_type_4_sx, x="time", y=["S_x", "S_y"])
    st.line_chart(dframe_incoherent_type_4, x="time", y=Plot_Control())
with tab_data:
    dframe_incoherent_type_4