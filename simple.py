import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# werte
Sx = 0
Sy = 1

x = 1
x_act = 0

y = 0
y_act = 0

z = 0

Kxz = 0.1
Kxy = 0.1
Kyz = 0.5 # 0.5, 5

By = 0
Bz = 0

betaY = 1
betaZ = 1

alphaY = 1
alphaZ = 1

H = 2

# Sx ON/OFF
def switch_Sx():
    if Sx == 1:
        return x_act = x


# Hill Funktion Pattern

def f_act(u, K, H):
    return (u / K) ** H / (1 + (u / K) ** H)

def f_rep(u, K, H):
    return (1 / (1 + (u / K) ** H))

def fc_act(u, Ku, Kv, v, H):
    return (u / Ku) ** H / (1 + (u / Ku) ** H + (v / Kv) ** H)

def fc_rep(u, Ku, Kv, v, H):
    return (1 / (1 + (u / Ku) ** H + (v / Kv) ** H))

# Coherent 1

def dYdt(t, Y, X_star, Kxy, ay, By, by, H, Sx, option):
    return By + by * f_act(X_star, Kxy, H) - ay * Y

def dZdt(t, Z, X_star, Kxz, Y_star, Kyz, az, Bz, bz, H, Sx, option):
    G_z = f_act(X_star_effect, Kxz, H) * f_act(X_star_effect, Kyz, H)
    return Bz + bz * G_z - az * Z

# system das übergben wird

def system(t, variables, Kxy, Kxz, Kyz, ay, By, by, az, Bz, bz, H, Sx, option, ):
    X_star, Y, Z = variables
    dXdt = 0  # Angenommen, X wird konstitutiv ausgedrückt
    dYdt_val = dYdt(t, Y, X_star, Kxy, ay, By, by, H, Sx, option)
    dZdt_val = dZdt(t, Z, X_star, Kxz, Y, Kyz, az, Bz, bz, H, Sx, option)
    return [dXdt, dYdt_val, dZdt_val]
