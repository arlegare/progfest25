import numpy as np

def modele_thermique(T, t, C, L, K_ext, T_ext_fct, P_fct):
    """
    Équation différentielle du modèle (écrite sous forme matricielle).
    
    Arguments d'entrée :
    T : Vecteur des températures.
    t : Temps "actuel".
    C : Matrice de capacité thermique.
    L : Matrice de conductivité thermique (laplacienne).
    K_ext : Matrice de conductivité thermique avec l'extérieur.
    T_ext_fct : Fonction pour le vecteur des températures extérieures (par ex. : un mur au soleil sera plus chaud).
    P_fct : Fonction de modulation des puissances selon les températures.
    
    Renvoie en sortie :
    - dTdt : Approximation discrète de la dérivée temporelle du vecteur des températures. 
    """

    T_ext = T_ext_fct(t)
    P = P_fct(t, T)
    dTdt = np.linalg.inv(C) @ (- L @ T  + P - K_ext @ (T - T_ext))

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
            P[i] = P_fct(t[i], T[i-1])
            T[i] = rk45_step(modele_thermique, T[i-1], t[i], dt, C, L, K_ext, T_ext_fct, P_fct)
    else:
        for i in range(1, len(t)):
            P[i] = P_fct(t[i], T[i-1])
            dTdt = modele_thermique(T[i-1], t[i], C, L, K_ext, T_ext_fct, P_fct)
            T[i] = T[i-1] + dTdt * dt

    return T, P

class PIDController:
    """
    Classe pour implémenter un contrôleur PID (similaire à un thermostat).

    Attributs :
    - T_target : Température cible.
    - K_p : Gain proportionnel.
    - K_i : Gain intégral.
    - K_d : Gain dérivatif.
    - integral : Intégrale accumulée de l'erreur.
    - previous_error : Erreur du pas de temps précédent.

    Méthodes :
    - update_T_target(T_cible) : Met à jour la température cible.
    - calcul_puissance(T, dt) : Calcule la puissance de chauffage.
    """

    def __init__(self, T_target, K_p=50.0, K_i=1.0, K_d=10.0):
        self.T_target = np.array(T_target, dtype=float)
        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d
        self.integral = np.zeros_like(self.T_target, dtype=float)
        self.previous_error = np.zeros_like(self.T_target, dtype=float)

    def update_T_target(self, T_cible):
        """Met à jour la température cible pour prendre en compte les changements dynamiques."""
        self.T_target = np.array(T_cible, dtype=float)

    def calcul_puissance(self, T, dt):
        """Calcule la puissance de chauffage à partir de l'erreur de température."""
        error = self.T_target - T
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        self.previous_error = error

        P = self.K_p * error + self.K_i * self.integral + self.K_d * derivative
        return np.maximum(P, 0)  # Pas de refroidissement !
