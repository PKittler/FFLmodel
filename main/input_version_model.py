import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def get_user_selection():
    # Coherent AND
    print("Bitte wählen Sie eine der folgenden Optionen für die dZdt-Funktion:")
    print("1: AND: X aktiviert Y, X aktiviert Z, Y aktiviert Z") # AAA
    print("2: AND: X inhibiert Y, X inhibiert Z, Y aktiviert Z") # IIA
    print("3: AND: X inhibiert Y, X aktiviert Z, Y inhibiert Z") # IAI
    print("4: AND: X aktiviert Y, X inhibiert Z, Y inhibiert Z") # AII
    # Incoherent AND
    print("5: AND: X aktiviert Y, X aktiviert Z, Y inhibiert Z") # AAI
    print("6: AND: X inhibiert Y, X inhibiert Z, Y inhibiert Z") # III
    print("7: AND: X inhibiert Y, X aktiviert Z, Y aktiviert Z") # IAA
    print("8: AND: X aktiviert Y, X inhibiert Z, Y aktiviert Z") # AIA
    # Coherent OR
    print("9: OR: X aktiviert Y, X aktiviert Z, Y aktiviert Z") # AAA
    print("10: OR: X inhibiert Y, X inhibiert Z, Y aktiviert Z") # IIA
    print("11: OR: X inhibiert Y, X aktiviert Z, Y inhibiert Z") # IAI
    print("12: OR: X aktiviert Y, X inhibiert Z, Y inhibiert Z") # AII
    # Incoherent OR
    print("13: OR: X aktiviert Y, X aktiviert Z, Y inhibiert Z") # AAI
    print("14: OR: X inhibiert Y, X inhibiert Z, Y inhibiert Z") # III
    print("15: OR: X inhibiert Y, X aktiviert Z, Y aktiviert Z") # IAA
    print("16: OR: X aktiviert Y, X inhibiert Z, Y aktiviert Z") # AIA
    while True:
        user_input = input("Geben Sie die Nummer der gewünschten Option ein (1-16): ")
        if 1 <= int(user_input) <= 16:
            return int(user_input)
        else:
            print("Ungültige Eingabe. Bitte wählen Sie eine Nummer zwischen 1 und 16.")

option = get_user_selection()

# Initialwerte und Params
Sx = 0

Kxy = 0.1 
Kxz = 0.1 
Kyz = 0.1

By = 0
Bz = 0

ay = 1  
az = 1 
 
by = 1 
bz = 1

H = 2

X_star = 1
Y0 = 0
Z0 = 0
Initialwerte = [X_star, Y0, Z0]

t_end = 20  
num_points = 200 


def f_act(u, K, H):
    return (u / K) ** H / (1 + (u / K) ** H)

def f_rep(u, K, H):
    return (1 / (1 + (u / K) ** H))

def fc_act(u, Ku, Kv, v, H):
    return (u / Ku) ** H / (1 + (u / Ku) ** H + (v / Kv) ** H)

def fc_rep(u, Ku, Kv, v, H):
    return (1 / (1 + (u / Ku) ** H + (v / Kv) ** H))




def dYdt(t, Y, X_star, Kxy, ay, By, by, H, Sx):
    X_star_effect = Sx * X_star  # Ein- und Ausschalten von Sx
    if option == 1 or 4 or 5 or 8 or 9 or 12 or 13 or 16:
        return By + by * f_act(X_star_effect, Kxy, H) - ay * Y
    else:
        return By + by * f_rep(X_star_effect, Kxy, H) - ay * Y
    


def dZdt(t, Z, X_star, Kxz, Y_star, Kyz, az, Bz, bz, H, Sx, option):
    X_star_effect = Sx * X_star  # Ein- und Ausschalten von Sx
    G_z = 0

    # AND: X activates Z, Y activates Z
    if option == 1 or 7: 
        G_z = f_act(X_star_effect, Kxz, H) * f_act(X_star_effect, Kyz, H)

    # AND: X inhibits Z, Y activates Z
    elif option == 2 or 8:
        G_z = f_rep(X_star_effect, Kxz, H) * f_act(X_star_effect, Kyz, H)

    # AND: X activates Z, Y inhibits Z
    elif option == 3 or 5:
        G_z = f_act(X_star_effect, Kxz, H) * f_rep(X_star_effect, Kyz, H)

    # AND: X inhibits Z. Y inhibits Z
    elif option == 4 or 6:
        G_z = f_rep(X_star_effect, Kxz, H) * f_rep(X_star_effect, Kyz, H)

    # OR: X activates Z, Y activates Z    
    elif option == 9 or 15:
        G_z = fc_act(X_star_effect, Kxz, Kyz, Y_star, H) + fc_act(Y_star, Kyz, Kxz, X_star_effect, H)

    # OR: X inhibits Z, Y activates Z
    elif option == 10 or 16:
        G_z = fc_rep(X_star_effect, Kxz, Kyz, Y_star, H) + fc_act(Y_star, Kyz, Kxz, X_star_effect, H)

    # OR: X activates Z, Y inhibits Z
    elif option == 11 or 13:
        G_z = fc_act(X_star_effect, Kxz, Kyz, Y_star, H) + fc_rep(Y_star, Kyz, Kxz, X_star_effect, H)

    # OR: X inhibits Z. Y inhibits Z
    elif option == 12 or 143:
       G_z = fc_rep(X_star_effect, Kxz, Kyz, Y_star, H) + fc_rep(Y_star, Kyz, Kxz, X_star_effect, H)

    return Bz + bz * G_z - az * Z




def system(t, variables, Kxy, Kxz, Kyz, ay, By, by, az, Bz, bz, H, Sx, option):
    X_star, Y, Z = variables
    dXdt = 0  # Angenommen, X wird konstitutiv ausgedrückt
    dYdt_val = dYdt(t, Y, X_star, Kxy, ay, By, by, H, Sx)
    dZdt_val = dZdt(t, Z, X_star, Kxz, Y, Kyz, az, Bz, bz, H, Sx, option)
    return [dXdt, dYdt_val, dZdt_val]




# Listen um X*, Y und Z zu speichern
X_star_values = []
Y_values = []
Z_values = []
Sx_values = []

# Liste von Zeiten entsprechend der Anzahl der Zeitschritte
t_values = np.linspace(0, t_end, num_points)

# Schaltzeiten für Sx
Zeit_switch = [5, 10, 20]



# Schleife über die Zeitschritte und Ergebniss berechnen
for t_step in range(num_points):
    t = t_step * t_end / num_points

    Sx_values.append(Sx)  # Sx in Liste speichern

    # Überprüfen, ob es Zeit ist, Sx ein- oder auszuschalten
    if t in Zeit_switch:
        Sx = 1 - Sx  # Wechseln zwischen 0 und 1

    sol = solve_ivp(
        system,
        (t, t + t_end / num_points),
        Initialwerte,
        args=(Kxy, Kxz, Kyz, ay, By, by, az, Bz, bz, H, Sx, option),
        t_eval=[t + t_end / num_points],  # Nur den letzten Zeitschritt auswerten
    )

    Initialwerte = [sol.y[0][-1], sol.y[1][-1], sol.y[2][-1]]  # Aktualisieren der Anfangsbedingungen

    X_star_values.append(sol.y[0][0])  # Nur den ersten Wert aus dem letzten Zeitschritt speichern
    Y_values.append(sol.y[1][0])  # Nur den ersten Wert aus dem letzten Zeitschritt speichern
    Z_values.append(sol.y[2][0])  # Nur den ersten Wert aus dem letzten Zeitschritt speichern












# Plot der Ergebnisse in einem gemeinsamen Plot
plt.figure(figsize=(12, 8))

# Erster Subplot: Sx Ein und Ausschalten
plt.subplot(2, 1, 1)
plt.plot(t_values, Sx_values, label='Sx')
plt.xlabel('Zeit')
plt.ylabel('Sx')
plt.title('Sx Ein- und Ausschalten')
plt.grid(True)

# Zweiter Subplot: Konzentrationen von X*, Y und Z
plt.subplot(2, 1, 2)
plt.plot(t_values, X_star_values, label='X*')
plt.plot(t_values, Y_values, label='Y')
plt.plot(t_values, Z_values, label='Z')
plt.xlabel('Zeit')
plt.ylabel('Konzentration')
plt.legend()

option_descriptions = {
    1: "AND: X → Y, X → Z, Y → Z",
    2: "AND: X ⊣ Y, X ⊣ Z, Y → Z",
    3: "AND: X ⊣ Y, X → Z, Y ⊣ Z",
    4: "AND: X → Y, X ⊣ Z, Y ⊣ Z",
    5: "AND: X → Y, X → Z, Y ⊣ Z",
    6: "AND: X ⊣ Y, X ⊣ Z, Y ⊣ Z", # III
    7: "AND: X ⊣ Y, X → Z, Y → Z", # IAA
    8: "AND: X → Y, X ⊣ Z, Y → Z", # AIA

    9: "OR: X → Y, X → Z, Y → Z", # AAA
    10: "OR: X ⊣ Y, X ⊣ Z, Y → Z", # IIA
    11: "OR: X ⊣ Y, X → Z, Y ⊣ Z", # IAI
    12: "OR: X → Y, X ⊣ Z, Y ⊣ Z", # AII
    13: "OR: X → Y, X → Z, Y ⊣ Z", # AAI
    14: "OR: X ⊣ Y, X ⊣ Z, Y ⊣ Z", # III
    15: "OR: X ⊣ Y, X → Z, Y → Z", # IAA
    16: "OR: X → Y, X ⊣ Z, Y → Z", # AIA    
}
plot_title = option_descriptions[option]
plt.title(plot_title)
plt.grid(True)

plt.tight_layout()  # Verbessert die Anordnung der Subplots, damit sie nicht überlappen.

plt.show()