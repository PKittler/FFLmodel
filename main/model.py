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

#st.set_page_config(page_title="Modelling a feed-forward loop network motif", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)
#st.title("Modelling a feed-forward loop network motif | M. Hirsch, P. Kittler")

st.sidebar.write("Show graphs:")
sb_showgraph_colx, sb_showgraph_coly, sb_showgraph_colz = st.sidebar.columns(3)
with sb_showgraph_colx:
    show_x_active = st.checkbox('X*', value=True)
with sb_showgraph_coly:
    show_y = st.checkbox('Y', value=False)
with sb_showgraph_colz:
    show_z = st.checkbox('Z', value=True)

if "visibility" not in st.session_state:
    st.session_state.horizontal = True

# parameters
x_active = 0
y_0 = 0
z_0 = 0

st.sidebar.subheader("Parameters")

c_logicmode = st.sidebar.radio("Coherent Z Logic Mode", ["AND", "OR"], horizontal=st.session_state.horizontal)

k_xy = st.sidebar.slider("K_xy", min_value=0.01, max_value=5., value=0.1, step=0.01, key="kxy")   # activation / repression coefficient XY
k_xz = st.sidebar.slider("K_xz", min_value=0.01, max_value=5., value=0.1, step=0.01)   # activation / repression coefficient XZ
k_yz = st.sidebar.slider("K_yz", min_value=0.01, max_value=5., value=0.5, step=0.01)   # activation / repression coefficient YZ

a_y = st.sidebar.slider("a_y", min_value=0.01, max_value=1., value=1., step=0.01)   # decay rate
a_z = st.sidebar.slider("a_z", min_value=0.01, max_value=1., value=1., step=0.01)   # decay rate

b_y = st.sidebar.slider("B_y", min_value=0., max_value=10., value=0., step=0.1)     # basal concentration of Y
b_z = st.sidebar.slider("B_z", min_value=0., max_value=10., value=0., step=0.1)     # basal concentration of Z

beta_y = st.sidebar.slider("beta_y", min_value=0., max_value=1., value=1., step=0.01)
beta_z = st.sidebar.slider("beta_z", min_value=0., max_value=1., value=1., step=0.01)

h = st.sidebar.slider("H", min_value=0, max_value=3, value=2)                       # exponent of Hill function

t_interval = st.sidebar.slider("Inducer active time interval", 1.0, 20.0, 10.)         # time endpoint
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

    return output

# function for regulation, named f in paper
def Regulation_Activator (u, k, h):
    return (((u / k)**h) / (1 + (u / k) ** h))

def Regulation_Repressor (u, k, h):
    return (1 / (1 + (u / k) ** h))

# function for competitive regulation, named f_c in paper
def Competitive_Regulation_Activator(u, v, k_u, k_v, h):
    return (((u / k_u) ** h) / (1 + (u / k_u) ** h + (v / k_v) ** h))

def Competitive_Regulation_Repressor(u, v, k_u, k_v, h):
    return (1/ (1 + (u / k_u) ** h + (v / k_v) ** h))

def C_Gate(x, y, k_xz, k_yz, h, activator_xz, activator_yz):

    # AND mode
    if(c_logicmode == "AND"):
        if activator_xz == True and activator_yz == True:
            return Regulation_Activator(x, k_xz, h) * Regulation_Activator(y, k_yz, h)
        if activator_xz == True and activator_yz == False:
            return Regulation_Activator(x, k_xz, h) * Regulation_Repressor(y, k_yz, h)
        if activator_xz == False and activator_yz == True:
            return Regulation_Repressor(x, k_xz, h) * Regulation_Activator(y, k_yz, h)
        if activator_xz == False and activator_yz == False:
            return Regulation_Repressor(x, k_xz, h) * Regulation_Repressor(y, k_yz, h)
    # OR mode
    elif(c_logicmode == "OR"):
        if activator_xz == True and activator_yz == True:
            return Competitive_Regulation_Activator(x, y, k_xz, k_yz, h) + Competitive_Regulation_Activator(y, x, k_yz, k_xz, h)
        if activator_xz == True and activator_yz == False:
            return Competitive_Regulation_Activator(x, y, k_xz, k_yz, h) + Competitive_Regulation_Repressor(y, x, k_yz, k_xz, h)
        if activator_xz == False and activator_yz == True:
            return Competitive_Regulation_Repressor(x, y, k_xz, k_yz, h) + Competitive_Regulation_Activator(y, x, k_yz, k_xz, h)
        if activator_xz == False and activator_yz == False:
            return Competitive_Regulation_Repressor(x, y, k_xz, k_yz, h) + Competitive_Regulation_Repressor(y, x, k_yz, k_xz, h)

def I_Gate(x, y, k_xz, k_yz, h, activator_xz, activator_yz):
    if activator_xz == True and activator_yz == True:
        return Regulation_Activator(x, k_xz, h) * Regulation_Activator(y, k_yz, h)
    if activator_xz == True and activator_yz == False:
        return Regulation_Activator(x, k_xz, h) * Regulation_Repressor(y, k_yz, h)
    if activator_xz == False and activator_yz == True:
        return Regulation_Repressor(x, k_xz, h) * Regulation_Activator(y, k_yz, h)
    if activator_xz == False and activator_yz == False:
        return Regulation_Repressor(x, k_xz, h) * Regulation_Repressor(y, k_yz, h)


# function for time derivative of Y
def C_ODE_Y (t, y, x_active, k_xy, a_y, b_y, beta_y, h, s_x, activator):
    x = x_active             # switch for turn X on/off

    if activator == True:
        return b_y + beta_y * Regulation_Activator(x, k_xy, h) - a_y * y
    else:
        return b_y + beta_y * Regulation_Repressor(x, k_xy, h) - a_y * y

# function for time derivative of Z
def C_ODE_Z (t, z, y,  x_active, k_xz, k_yz, a_z, b_z, beta_z, h, s_x, activator_xz, activator_yz):
    x = x_active              # switch for turn X on/off
    return b_z + beta_z * C_Gate(x, y, k_xz, k_yz, h, activator_xz, activator_yz) - a_z * z

# function for time derivative of Y
def I_ODE_Y (t, y, x_active, k_xy, a_y, b_y, beta_y, h, s_x, activator):
    x = x_active             # switch for turn X on/off

    if activator == True:
        return b_y + beta_y * Regulation_Activator(x, k_xy, h) - a_y * y
    else:
        return b_y + beta_y * Regulation_Repressor(x, k_xy, h) - a_y * y

# function for time derivative of Z
def I_ODE_Z (t, z, y,  x_active, k_xz, k_yz, a_z, b_z, beta_z, h, s_x, activator_xz, activator_yz):
    x = x_active              # switch for turn X on/off
    return b_z + beta_z * I_Gate(x, y, k_xz, k_yz, h, activator_xz, activator_yz) - a_z * z

# function for model to merge ODE_Y and ODE_Z to a ODE system
def CT_Model (t, init_vars, k_xy, k_xz, k_yz, a_y, b_y, beta_y, a_z, b_z, beta_z, h, s_x, s_y, activator_xz, activator_xy, activator_yz):
    x_active, y, z = init_vars
    x_active = s_x

    dxdt = 0                        # if X is constant, then change rate is 0
    dydt = C_ODE_Y(t, y, x_active, k_xy, a_y, b_y, beta_y, h, s_x, activator_xy)
    dzdt = C_ODE_Z(t, z, y, x_active, k_xz, k_yz, a_z, b_z, beta_z, h, s_x, activator_xz, activator_yz)

    return [dxdt, dydt, dzdt]

def IT_Model (t, init_vars, k_xy, k_xz, k_yz, a_y, b_y, beta_y, a_z, b_z, beta_z, h, s_x, s_y, activator_xz, activator_xy, activator_yz):
    x_active, y, z = init_vars
    x_active = s_x

    dxdt = 0                        # if X is constant, then change rate is 0
    dydt = I_ODE_Y(t, y, x_active, k_xy, a_y, b_y, beta_y, h, s_x, activator_xy)
    dzdt = I_ODE_Z(t, z, y, x_active, k_xz, k_yz, a_z, b_z, beta_z, h, s_x, activator_xz, activator_yz)

    return [dxdt, dydt, dzdt]


def CalculateCType(s_x, s_y, t_end, activator_xz, activator_xy, activator_yz):

    initial_values = [1, b_y, b_z]

    x_active_values = []
    y_values = []
    z_values = []


    s_x_values = []
    s_y_values = []

    t_span = t_end
    t_values = np.linspace(0, 20, steps)

    for t_step in range(steps):
        t = (20 / steps) * t_step

        if (t <= t_end):
            s_x_values.append(s_x)  # write s_x in list
            s_y_values.append(s_y)  # write s_y in list
        else:
            s_x_values.append(0)  # write s_x in list
            s_y_values.append(0)  # write s_y in list


        model_solver = solve_ivp(CT_Model, (t, t + t_span / steps), initial_values, args=(k_xy, k_xz, k_yz, a_y, b_y, beta_y, a_z, b_z, beta_z, h, s_x_values[-1], s_y_values[-1], activator_xz, activator_xy, activator_yz), t_eval=[t + t_span / steps])

        # update initial value (conditions)
        initial_values = [model_solver.y[0][-1], model_solver.y[1][-1], model_solver.y[2][-1]]

        x_active_values.append(model_solver.y[0][0])  # save only first value from last step
        y_values.append(model_solver.y[1][0])  # save only first value from last step
        z_values.append(model_solver.y[2][0])  # save only first value from last step

    # pack the results in 1 output object
    output_object = [t_values, x_active_values, y_values, z_values, s_x_values, s_y_values]

    return output_object

def CalculateIType(s_x, s_y, t_end, activator_xz, activator_xy, activator_yz):

    initial_values = [1, b_y, b_z]

    x_active_values = []
    y_values = []
    z_values = []


    s_x_values = []
    s_y_values = []

    t_span = t_end
    t_values = np.linspace(0, 20, steps)

    for t_step in range(steps):
        t = (20 / steps) * t_step

        if (t <= t_end):
            s_x_values.append(s_x)  # write s_x in list
            s_y_values.append(s_y)  # write s_y in list
        else:
            s_x_values.append(0)  # write s_x in list
            s_y_values.append(0)  # write s_y in list


        model_solver = solve_ivp(IT_Model, (t, t + t_span / steps), initial_values, args=(k_xy, k_xz, k_yz, a_y, b_y, beta_y, a_z, b_z, beta_z, h, s_x_values[-1], s_y_values[-1], activator_xz, activator_xy, activator_yz), t_eval=[t + t_span / steps])

        # update initial value (conditions)
        initial_values = [model_solver.y[0][-1], model_solver.y[1][-1], model_solver.y[2][-1]]

        x_active_values.append(model_solver.y[0][0])  # save only first value from last step
        y_values.append(model_solver.y[1][0])  # save only first value from last step
        z_values.append(model_solver.y[2][0])  # save only first value from last step

    # pack the results in 1 output object
    output_object = [t_values, x_active_values, y_values, z_values, s_x_values, s_y_values]

    return output_object

# COHERENT TYPE 1
CT1 = CalculateCType(s_x = 1, s_y = 1, t_end = t_interval, activator_xz = True, activator_xy = True, activator_yz =True)

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

CT2 = CalculateCType(s_x = 0, s_y = 1, t_end = t_interval, activator_xz = False, activator_xy = False, activator_yz = True)

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

CT3 = CalculateCType(s_x = 1, s_y = 1, t_end = t_interval, activator_xz = False, activator_xy = True, activator_yz = False)

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

CT4 = CalculateCType(s_x = 1, s_y = 1, t_end = t_interval, activator_xz = True, activator_xy = False, activator_yz = False)

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
    st.line_chart(dframe_coherent_type_1_sx, x="time", y=["S_x", "S_y"], height=200)
    st.line_chart(dframe_coherent_type_1, x="time", y=Plot_Control())
with tab_data:
    dframe_coherent_type_1


with col_coherent_type_2:
    st.subheader("Coherent Type 2")
    tab_plot, tab_data = st.tabs(["Plot", "Data"])

with tab_plot:
    st.line_chart(dframe_coherent_type_2_sx, x="time", y=["S_x", "S_y"], height=200)
    st.line_chart(dframe_coherent_type_2, x="time", y=Plot_Control())
with tab_data:
    dframe_coherent_type_2


with col_coherent_type_3:
    st.subheader("Coherent Type 3")
    tab_plot, tab_data = st.tabs(["Plot", "Data"])

with tab_plot:
    st.line_chart(dframe_coherent_type_3_sx, x="time", y=["S_x", "S_y"], height=200)
    st.line_chart(dframe_coherent_type_3, x="time", y=Plot_Control())
with tab_data:
    dframe_coherent_type_3


with col_coherent_type_4:
    st.subheader("Coherent Type 4")
    tab_plot, tab_data = st.tabs(["Plot", "Data"])

with tab_plot:
    st.line_chart(dframe_coherent_type_4_sx, x="time", y=["S_x", "S_y"], height=200)
    st.line_chart(dframe_coherent_type_4, x="time", y=Plot_Control())
with tab_data:
    dframe_coherent_type_4


# INCOHERENT TYPE 1

IT1 = CalculateIType(s_x = 1, s_y = 1, t_end = t_interval, activator_xz = True, activator_xy = True, activator_yz = False)

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

IT2 = CalculateIType(s_x = 1, s_y = 1, t_end = t_interval, activator_xz = False, activator_xy = False, activator_yz = False)

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

IT3 = CalculateIType(s_x = 1, s_y = 1, t_end = t_interval, activator_xz = False, activator_xy = True, activator_yz = True)

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

IT4 = CalculateIType(s_x = 1, s_y = 1, t_end = t_interval, activator_xz = True, activator_xy = False, activator_yz = True)

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
    st.line_chart(dframe_incoherent_type_1_sx, x="time", y=["S_x", "S_y"], height=200)
    st.line_chart(dframe_incoherent_type_1, x="time", y=Plot_Control())
with tab_data:
    dframe_incoherent_type_1


with col_incoherent_type_2:
    st.subheader("Incoherent Type 2")
    tab_plot, tab_data = st.tabs(["Plot", "Data"])

with tab_plot:
    st.line_chart(dframe_incoherent_type_2_sx, x="time", y=["S_x", "S_y"], height=200)
    st.line_chart(dframe_incoherent_type_2, x="time", y=Plot_Control())
with tab_data:
    dframe_incoherent_type_2


with col_incoherent_type_3:
    st.subheader("Incoherent Type 3")
    tab_plot, tab_data = st.tabs(["Plot", "Data"])

with tab_plot:
    st.line_chart(dframe_incoherent_type_3_sx, x="time", y=["S_x", "S_y"], height=200)
    st.line_chart(dframe_incoherent_type_3, x="time", y=Plot_Control())
with tab_data:
    dframe_incoherent_type_3


with col_incoherent_type_4:
    st.subheader("Incoherent Type 4")
    tab_plot, tab_data = st.tabs(["Plot", "Data"])

with tab_plot:
    st.line_chart(dframe_incoherent_type_4_sx, x="time", y=["S_x", "S_y"], height=200)
    st.line_chart(dframe_incoherent_type_4, x="time", y=Plot_Control())
with tab_data:
    dframe_incoherent_type_4