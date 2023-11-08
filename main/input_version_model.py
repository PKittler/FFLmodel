import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def get_user_selection():
    print("Bitte wählen Sie eine der folgenden Optionen für die dZdt-Funktion:")
    print("1: AND: X aktiviert Z, Y aktiviert Z")
    print("2: AND: X inhibiert Z, Y aktiviert Z")
    print("3: AND: X aktiviert Z, Y inhibiert Z")
    print("4: AND: X inhibiert Z, Y inhibiert Z")
    print("5: OR: X aktiviert Z, Y aktiviert Z")
    print("6: OR: X inhibiert Z, Y aktiviert Z")
    print("7: OR: X aktiviert Z, Y inhibiert Z")
    print("8: OR: X inhibiert Z, Y inhibiert Z")
    while True:
        user_input = input("Geben Sie die Nummer der gewünschten Option ein (1-8): ")
        if user_input.isdigit() and 1 <= int(user_input) <= 8:
            return int(user_input)
        else:
            print("Ungültige Eingabe. Bitte wählen Sie eine Nummer zwischen 1 und 8.")

option = get_user_selection()

# Initialwerte und Params
Sx = 0

Kxy = 1 
Kxz = 1 
Kyz = 1 

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
    return By + by * f_act(X_star_effect, Kxy, H) - ay * Y


def dZdt(t, Z, X_star, Kxz, Y_star, Kyz, az, Bz, bz, H, Sx, option):
    X_star_effect = Sx * X_star  # Ein- und Ausschalten von Sx
    G_z = 0
    # AND: X activates Z, Y activates Z
    if option == 1: 
        G_z = f_act(X_star_effect, Kxz, H) * f_act(X_star_effect, Kyz, H)
    # AND: X represses Z, Y activates Z
    elif option == 2:
        G_z = f_rep(X_star_effect, Kxz, H) * f_act(X_star_effect, Kyz, H)
    # AND: X activates Z, Y represses Z
    elif option == 3:
        G_z_AND = f_act(X_star_effect, Kxz, H) * f_rep(X_star_effect, Kyz, H)
    # AND: X represses Z. Y represses Z
    elif option == 4:
        G_z_AND = f_rep(X_star_effect, Kxz, H) * f_rep(X_star_effect, Kyz, H)
    # OR: X activates Z, Y activates Z    
    elif option == 5:
        G_z_OR = fc_act(X_star_effect, Kxz, Kyz, Y_star, H) + fc_act(Y_star, Kyz, Kxz, X_star_effect, H)
    # OR: X represses Z, Y activates Z
    elif option == 6:
        G_z_OR = fc_rep(X_star_effect, Kxz, Kyz, Y_star, H) + fc_act(Y_star, Kyz, Kxz, X_star_effect, H)
    # OR: X activates Z, Y represses Z
    elif option == 7:
        G_z_OR = fc_act(X_star_effect, Kxz, Kyz, Y_star, H) + fc_rep(Y_star, Kyz, Kxz, X_star_effect, H)
    # OR: X represses Z. Y represses Z
    elif option == 8:
       G_z_OR = fc_rep(X_star_effect, Kxz, Kyz, Y_star, H) + fc_rep(Y_star, Kyz, Kxz, X_star_effect, H)

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
Zeit_switch = [5, 10, 15]



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



option_descriptions = {
    1: "AND: X aktiviert Z, Y aktiviert Z",
    2: "AND: X inhibiert Z, Y aktiviert Z",
    3: "AND: X aktiviert Z, Y inhibiert Z",
    4: "AND: X inhibiert Z, Y inhibiert Z",
    5: "OR: X aktiviert Z, Y aktiviert Z",
    6: "OR: X inhibiert Z, Y aktiviert Z",
    7: "OR: X aktiviert Z, Y inhibiert Z",
    8: "OR: X inhibiert Z, Y inhibiert Z"
}





# Setzen Sie den Plot-Titel basierend auf der gewählten Option
plot_title = option_descriptions[option]


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
plt.title(plot_title)
plt.grid(True)

plt.tight_layout()  # Verbessert die Anordnung der Subplots, damit sie nicht überlappen.

plt.show()


