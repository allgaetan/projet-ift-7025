# Projet: IFT-7025, Automne 2025

**Equipe 55:** Gaétan ALLAIRE

## Réseaux de neurones pour l’apprentissage par renforcement dans le jeu de Pacman

Dans ce projet, nous ajoutons un réseau de neurones perceptron multi-couches (MLPRegressor de scikit-learn) à l'algorithme d'apprentissage par renforcement Pacman, notamment pour l'apprentissage de sa fonction Q. Nous commencerons par définir un nouveau feature extractor spécifique pour le MLPRegressor, puis un nouvel agent basé sur la logique du ApproximateQAgent, qui intégrera le MLPRegressor pour apprendre la fonction Q. Finalement, nous évaluerons notre modèle à travers différents types de tests. Une attention particulière sera porté sur l'étude de l'influence des paramètres du perceptron, notamment la régularisation du modèle, ainsi que sur sa capacité à généraliser son apprentissage.

>**NOTE:** Ce README présente seulement la structure et l'exécution du projet. Pour les détails sur la réalisation de ce projet ainsi que sur les résultats, merci de vous référer au [rapport](ift7025_project_report.pdf).

### Structure du projet

Ce projet reprend la totalité du code du TP2. Les seules parties ajoutées et modifées spécifiquement pour ce projet se trouvent dans les fichiers suivants:
- [nnFeatureExtractors.py](nnFeatureExtractors.py) pour la classe `MLPRegressorExtractor`, l'extracteur de features spécifique au MLPRegressor
- [nnQLearningAgents](nnQLearningAgents.py) pour la classe `MLPRegressorQAgent`, l'agent qui implémente le perceptron multi-couches MLPRegressor pour apprendre la fonction Q de Pacman
- [eval.py](eval.py) qui implémente les différentes fonctions qui ont permis d'évaluer l'agent et les hyperparamètres
- [pacman.py](pacman.py): la fonction `runGames()` a été légèrement modifiée pour pouvoir retourner les différents scores pour le besoin des métriques d'évaluation
- [README.md](README.md), ce fichier README
- [ift7025_project_report.pdf](ift7025_project_report.pdf), le rapport du projet
- [evalLogs.txt](evalLogs.txt): un fichier log se remplit lors des évaluations afin de garder des traces de certains résultats

### Train/Test:

Pour lancer vos propres ésisodes de train et de test, vous pouvez exécuter les commandes suivantes:

- Pour des mêmes paramètres entre train et test:
```
python pacman.py -p MLPRegressorQAgent -x NUM_TRAIN -n NUM_TRAIN + NUM_TEST -l GRID_NAME -g GHOST_TYPE (Default RandomGhost)
```

- Avec des paramètres différents entre train et test:
```
python pacman.py -p MLPRegressorQAgent -x NUM_TRAIN -n NUM_TRAIN -l GRID_NAME -g GHOST_TYPE (Default RandomGhost)
```
```
python pacman.py -p MLPRegressorQAgent -n NUM_TEST -l GRID_NAME -g GHOST_TYPE (Default RandomGhost)
```

### Exécuter le projet

Exécuter le fichier [eval.py](eval.py) effectuera tous les benchmarks exécutés pour l'évaluation. 
```
python eval.py -n NUM_TRAIN -m NUM_TEST -p -s -g -r -l
```
Les options pour l'exécution sont (avec `python eval.py -h`):

```
Evaluation methods for MLPRegressorQAgent

options:
  -h, --help            show this help message and exit
  -n NUM_TRAIN, --num_train NUM_TRAIN
                        Number of training games
  -m NUM_TEST, --num_test NUM_TEST
                        Number of testing games
  -p, --perform-hyperparams-evaluations
                        Perform hyperparameters evaluations
  -s, --perform-standard-evaluations
                        Perform standard evaluations
  -g, --perform-generalization-evaluations
                        Perform generalization evaluations
  -r, --perform-regularizations-evaluations
                        Perform regularizations evaluations
  -l, --logging         Enable logging
```