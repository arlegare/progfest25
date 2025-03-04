import numpy as np
from scipy.integrate import odeint

def modele_thermique(T, t, C, L, K_ext, T_ext_fct, P_fct):
    """
    Équation différentielle du modèle (écrite sous forme matricielle).
    
    Arguments d'entrée :
    T : Vecteur des températures;
    t : Temps "actuel";
    C : Matrice de capacité thermique;
    L : Matrice de conductivité thermique (laplacienne);
    K_ext : Matrice de conductivité thermique avec l'extérieur;
    T_ext_fct : Fonction pour le vecteur des températures extérieures (par ex. : un mur au soleil sera plus chaud);
    P_fct : Fonction de modulation des puissances selon les températures.
    
    Renvoie en sortie :
    - dTdt : Approximation discrète de la dérivée temporelle du vecteur des températures. 
    """

    T_ext = T_ext_fct(t)
    P = P_fct(t, T)
    dTdt = np.linalg.inv(C) @ (- L @ T  + P + K_ext @ (T_ext - T))

    return dTdt



def simule_modele_thermique(C, L, K_ext, T_ext_fct, P_fct, T_0, t, method="RK45"):
    """
    Simule la dynamique thermique d'un bâtiment à N pièces.
    Utilise la fonction 
    
    Arguments d'entrée :
    C : Matrice de capacités thermiques.
    L : Matrice de conductance thermique.
    K_ext : Matrice de conductance thermique externe.
    T_ext_fct : Fonction de température extérieure.
    P_fct : Fonction de puissance de chauffage.
    T_0 : Températures initiales.
    t : Vecteur temporel pour l'intégration.
    
    Renvoie en sortie :
    T : Températures des pièces au cours du temps.
    P : Puissances de chauffage au cours du temps.
    """

    dt = t[1] - t[0] # Approximation discrète de la dérivée temporelle du vecteur des températures.
    T = np.zeros((len(t), len(T_0)))
    P = np.zeros((len(t), len(T_0)))
    T[0] = T_0 # Conditions initiales.
    
    # Intégration Runge-Kutta 4(5) par défaut, sinon méthode d'Euler.
    if method == "RK45":
        def rk45_step(f, y, t, dt, *args):
            k1 = f(y, t, *args)
            k2 = f(y + dt/2 * k1, t + dt/2, *args)
            k3 = f(y + dt/2 * k2, t + dt/2, *args)
            k4 = f(y + dt * k3, t + dt, *args)
            return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        for i in range(1, len(t)):
            T[i] = rk45_step(modele_thermique, T[i-1], t[i], dt, C, L, K_ext, T_ext_fct, P_fct)
    else:
        for i in range(1, len(t)):
            dTdt = modele_thermique(T[i-1], t[i], C, L, K_ext, T_ext_fct, P_fct)
            T[i] = T[i-1] + dTdt * dt
    
    return T, P