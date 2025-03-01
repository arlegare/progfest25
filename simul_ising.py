import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os


# Paramètres
L = 20  # Taille de la grille (LxL)
T = 2.5  # Température
J = 1  # Force d'interaction (J > 0 pour un modèle ferromagnétique)
steps = 10000  # Nombre d'itérations

# Fonction pour calculer l'énergie d'une configuration de spins
def calculate_energy(spins, J):
    energy = 0
    for i in range(L):
        for j in range(L):
            # Interaction avec les voisins (périodicité aux bords)
            neighbors = spins[(i+1)%L, j] + spins[i, (j+1)%L] + spins[(i-1)%L, j] + spins[i, (j-1)%L]
            energy -= J * spins[i, j] * neighbors
    return energy / 2  # Chaque interaction a été comptée deux fois

# Fonction pour mettre à jour la configuration de spins
def metropolis(spins, T, J):
    # Choisir un spin aléatoire
    i, j = np.random.randint(0, L, 2)

    # Calculer l'énergie avant et après le flip du spin
    current_spin = spins[i, j]
    neighbors = spins[(i+1)%L, j] + spins[i, (j+1)%L] + spins[(i-1)%L, j] + spins[i, (j-1)%L]
    delta_E = 2 * J * current_spin * neighbors

    # Probabilité d'accepter le flip
    if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
        spins[i, j] = -spins[i, j]

# Initialisation de la grille avec des spins aléatoires
spins = np.random.choice([-1, 1], size=(L, L))

# Simulation de Metropolis
energy_list = []
for step in range(steps):
    metropolis(spins, T, J)
    if step % 100 == 0:  # Enregistrer l'énergie tous les 100 pas
        energy = calculate_energy(spins, J)
        energy_list.append(energy)

# Visualisation du résultat
plt.figure(figsize=(8, 6))
plt.imshow(spins, cmap='coolwarm', interpolation='nearest')
plt.title(f"Configuration des spins à T = {T}")
plt.colorbar(label="Spin")
plt.show()

# Tracer l'évolution de l'énergie
plt.figure(figsize=(8, 6))
plt.plot(energy_list)
plt.title("Évolution de l'énergie au cours de la simulation")
plt.xlabel("Pas de temps")
plt.ylabel("Énergie")
plt.show()
