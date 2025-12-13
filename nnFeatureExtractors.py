import numpy as np
from featureExtractors import FeatureExtractor, Directions, util
from util import manhattanDistance

class NeuralNetworkExtractor(FeatureExtractor):
    def __init__(self):
        self._feature_names = None

    def getFeatures(self, state, action):
        features = util.Counter()

        pacmanPos = state.getPacmanPosition()
        walls = state.getWalls()
        food = state.getFood()
        numFood = state.getNumFood()
        capsules = state.getCapsules()
        ghostsPos = state.getGhostPositions()
        ALL_ACTIONS = [
            Directions.NORTH, 
            Directions.SOUTH, 
            Directions.EAST, 
            Directions.WEST, 
            Directions.STOP
        ]

        pacmanX, pacmanY = float(pacmanPos[0]), float(pacmanPos[1])
        width, height = walls.width, walls.height
        foodList = food.asList()
        nearestFoodDistance = min(manhattanDistance(pacmanPos, foodPos) for foodPos in foodList) if foodList else 0.0
        nearestCapsuleDistance = min(manhattanDistance(pacmanPos, capPos) for capPos in capsules) if capsules else 0.0    
        maxDistance = float(width + height)

        features["pacman_x_norm"] = pacmanX / (width - 1.0)
        features["pacman_y_norm"] = pacmanY / (height - 1.0)
        for a in ALL_ACTIONS:
            features[f"action_{a}"] = 1.0 if a == action else 0.0
        features["nearest_food_distance_norm"] = nearestFoodDistance / maxDistance
        features["num_food_norm"] = numFood / (width * height)
        features["nearest_capsule_distance_norm"] = nearestCapsuleDistance / maxDistance
        for i, ghostPos in enumerate(ghostsPos):
            ghostDistance = manhattanDistance(pacmanPos, ghostPos)
            features[f"ghost_{i}_distance_norm"] = ghostDistance / maxDistance
            features[f"ghost_{i}_is_close"] = 1.0 if ghostDistance <= 2 else 0.0

        return features
    
    def getFeatureNames(self, state, action):
        if self._feature_names is not None:
            return self._feature_names
        feat = self.getFeatures(state, action)
        self._feature_names = sorted(feat.keys())
        return self._feature_names

    def getFeatureVector(self, state, action):
        features = self.getFeatures(state, action)
        if self._feature_names is None:
            self._feature_names = sorted(features.keys())
        vec = np.array([features.get(name, 0.0) for name in self._feature_names], dtype=float)
        return vec