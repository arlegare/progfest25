# Atelier de simulation numérique - ProgFest 2025

## **Description**

Ce projet contient plusieurs exemples de simulations numériques destinées à illustrer des phénomènes physiques complexes. Chaque simulation explore un domaine spécifique de la physique, en utilisant des modèles mathématiques et numériques pour résoudre des problèmes qui ne peuvent généralement pas être abordés de manière analytique. Les simulations sont présentées sous forme de notebooks Jupyter, accompagnés de fichiers Python contenant le code principal.

### **Exemples de simulations inclues :**

1. **Dynamique écologique**  
   Modélisation de la coopération entre espèces et de son impact sur leur répartition spatiale. Ce modèle simple explore les interactions entre populations et l’évolution des dynamiques écologiques.  

2. **Modèle d’Ising en 2D**  
   Une simulation de transitions de phase dans un système de spins, utilisant l’algorithme de Metropolis-Hastings. Ce modèle est couramment utilisé pour comprendre les phénomènes de magnétisme et la théorie des transitions de phase.

3. **Échanges thermiques dans un bâtiment**  
   Simulation des échanges thermiques dans un bâtiment avec plusieurs pièces, pour tester différentes stratégies de gestion du chauffage en fonction de la durée d’absence. Cette simulation peut être appliquée dans le cadre de l’optimisation énergétique des bâtiments.

## **Structure du projet**

- `notebooks/` : Contient les notebooks Jupyter présentant chaque simulation et expliquant les équations et les principes physiques sous-jacents.
- `scripts/` : Contient les fichiers `.py` qui contiennent le code de calcul principal des simulations.
- `requirements.txt` : Liste des dépendances nécessaires à l’exécution du projet.
- `README.md` : Ce fichier de présentation.

## **Installation**

1. Clonez le dépôt sur votre machine locale :

```bash
git clone https://github.com/arlegare/progfest25.git
cd votre-repository
```

1. Installez les dépendances nécessaires :

```bash
pip install -r requirements.txt
```

## **Usage**
* Le notebook contient des instructions étape par étape pour comprendre et exécuter les simulations.
* Vous pouvez ajuster les paramètres du modèle dans les fichiers Python pour explorer différentes configurations et observer l’évolution des systèmes simulés.

