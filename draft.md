# Projet: IFT-7025, Automne 2025
Equipe 55

## Réseaux de neurones pour l’apprentissage par renforcement dans le jeu de Pacman

_Dans ce projet, votre travail consiste à modifier le code du TP2 afin d’utiliser un réseau de neurones
(perceptron multi-couches) pour apprendre la fonction Q de Pacman. Fondamentalement, vous
devez réécrire le code de la classe ApproximateQAgent. Voici comment procéder en trois étapes
fondamentales:_

- intro

### Définition de la représentation du jeu

_Un réseau de neurones prendra en entrée un vecteur x = ϕ(s), où s est l’état du jeu et ϕ est une
fonction qui convertit l’état du jeu sous une forme appropriée. C’est à vous de définir la fonction
ϕ. Décrivez dans votre rapport comment vous avez procédé. Avez-vous normalisé le vecteur?
Quelle est sa dimension? Contient-il essentiellement toute l’information sur le jeu ou seulement
des caractéristiques spécifiques que vous avez jugées utiles?_

[nnFeatureExtractors.py](nnFeatureExtractors.py)

- classe `MLPRegressorExtractor`
- liste des features avec explication
- dimension
- normalisation
- quelles info sont omises? pourquoi?

### Réseaux de neurones de scikit-learn

_Exploiter la classe MLPRegressor de scikit-learn pour réécrire la classe ApproximateQAgent du TP2
(fichier qlearningAgents.py). À noter que l’option warm_start = T rue permet de faire plusieurs
appels à la méthode fit d’une manière à continuer l’entraînement sur les poids précédemment
obtenus (style online)._

[nnQLearningAgents](nnQLearningAgents.py)

- explication rapide de la classe `MLPRegressorQAgent`

### Évaluation

_Choisir un protocole pour déterminer les hyperparamètres de l’algorithme. Parmi les hyperparamètres, on compte par exemple le taux d’apprentissage, le nombre de couches, le nombre
de neurones à chaque couche. Un choix standard pour les fonctions d’activations est la ReLU et
un choix standard pour l’optimiseur est SGD.
Tester votre méthode sur différentes grilles. Tester contre le RandomGhost (défaut) et contre le
DirectionalGhost (plus difficile). Comparer vos scores moyens avec ceux de la version originale du
TP2._

- gridsearchCV explication
- quels paramètres évalués?
- sur quels ranges?
- benchmark
- tableau des résultats, comparaison avec TP2

_Seulement IFT-7025: Investiguer la capacité à généraliser de votre agent Pacman. Par exemple,
entraîner Pacman sur une grille puis tester sur une grille différente. La performance chute-t-elle
significativement? À noter qu’il faut une même représentation x = ϕ(s) qui peut être utilisée sur
des grilles de différentes tailles. Un autre exemple consiste à entraîner contre le RandomGhost
puis tester contre le DirectionalGhost. Ensuite, investiguer si la régularisation aide à augmenter la
généralisation. Quelle forme de régularisation avez-vous considérée?_

- entrainement smallGrid, test mediumGrid
- entrainement mediumGrid, test smallGrid
- représentation ok si normalisée?
- entrainement RandomGhost, test DirectionalGhost
- same avec régularisation l2

- benchmarking final

### premiers resultats