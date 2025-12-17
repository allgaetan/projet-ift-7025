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

hyperparameters evaluation:
```
Logs from function hyperparametersEvaluations
Configuration: Base
Hyperparameters: {'hidden_layer_sizes': (32,), 'activation': 'relu', 'solver': 'sgd', 'learning_rate_init': 0.001, 'warm_start': True}
Score: 0.78

Logs from function hyperparametersEvaluations
Configuration: ConfigA
Hyperparameters: {'hidden_layer_sizes': (64,), 'activation': 'relu', 'solver': 'sgd', 'learning_rate_init': 0.001, 'warm_start': True}
Score: 0.76

Logs from function hyperparametersEvaluations
Configuration: ConfigB
Hyperparameters: {'hidden_layer_sizes': (32,), 'activation': 'logistic', 'solver': 'sgd', 'learning_rate_init': 0.001, 'warm_start': True}
Score: 0.72

Logs from function hyperparametersEvaluations
Configuration: ConfigC
Hyperparameters: {'hidden_layer_sizes': (32,), 'activation': 'relu', 'solver': 'adam', 'learning_rate_init': 0.001, 'warm_start': True}
Score: 0.08

Logs from function hyperparametersEvaluations
Configuration: ConfigD
Hyperparameters: {'hidden_layer_sizes': (32,), 'activation': 'relu', 'solver': 'sgd', 'learning_rate_init': 0.01, 'warm_start': True}
Score: 0.0

Logs from function hyperparametersEvaluations
Results:          Score
Base      0.78
ConfigA   0.76
ConfigB   0.72
ConfigC   0.08
ConfigD   0.00
```

standard evaluation:
```
Logs from function standardEvaluations
Grid: smallGrid
Score: 0.79

Logs from function standardEvaluations
Grid: mediumGrid
Score: 1.0

Logs from function standardEvaluations
Grid: largeGrid
Score: 1.0

Logs from function standardEvaluations
Results:             Score
smallGrid    0.79
mediumGrid   1.00
largeGrid    1.00
```

generalization evaluation
```
Logs from function generalizationEvaluations
Test Case: SmallToMedium
Train Grid: smallGrid
Train Ghost Type: RandomGhost
Test Grid: mediumGrid
Test Ghost Type: RandomGhost
Score: 1.0

Logs from function generalizationEvaluations
Test Case: MediumToSmall
Train Grid: mediumGrid
Train Ghost Type: RandomGhost
Test Grid: smallGrid
Test Ghost Type: RandomGhost
Score: 0.78

Logs from function generalizationEvaluations
Test Case: RandomToDirectional
Train Grid: mediumGrid
Train Ghost Type: RandomGhost
Test Grid: mediumGrid
Test Ghost Type: DirectionalGhost
Score: 1.0

Logs from function generalizationEvaluations
Results:                      Score
SmallToMedium         1.00
MediumToSmall         0.78
RandomToDirectional   1.00
```

regularization evaluation
```
Logs from function regularizationsEvaluations
L2 Regularization Alpha: 0.0
Score: 0.76

Logs from function regularizationsEvaluations
L2 Regularization Alpha: 0.0001
Score: 0.78

Logs from function regularizationsEvaluations
L2 Regularization Alpha: 0.001
Score: 0.72

Logs from function regularizationsEvaluations
L2 Regularization Alpha: 0.01
Score: 0.7

Logs from function regularizationsEvaluations
L2 Regularization Alpha: 0.1
Score: 0.74

Logs from function regularizationsEvaluations
Results:         Score
0.0000   0.76
0.0001   0.78
0.0010   0.72
0.0100   0.70
0.1000   0.74
```

python eval.py -n 3000 -m 500 -s -g -r -l 