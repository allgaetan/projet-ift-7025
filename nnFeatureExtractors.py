import numpy as np
import util

from game import Actions
from featureExtractors import FeatureExtractor, closestFood

class MLPRegressorExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        pacmanPos = state.getPacmanPosition()
        numFood = state.getNumFood()
        
        features = util.Counter()

        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        pacmanX, pacmanY = float(pacmanPos[0]), float(pacmanPos[1])
        width, height = walls.width, walls.height
        dist = closestFood((next_x, next_y), food, walls)

        features["bias"] = 1.0
        features["pacman_x_norm"] = pacmanX / (width - 1.0)
        features["pacman_y_norm"] = pacmanY / (height - 1.0)
        features["num_food_norm"] = numFood / (width * height)
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
        if dist is not None:
            features["closest-food"] = float(dist) / (walls.width * walls.height)

        return features
    
    def toInputVector(self, features):
        featuresVector = np.array([features[key] for key in sorted(features.keys())])
        X = featuresVector.reshape(1, -1)
        return X
        