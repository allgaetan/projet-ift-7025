# featureExtractors.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"Feature extractors for Pacman game states"

from game import Directions, Actions
import util

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features

from util import manhattanDistance
import numpy as np

class CustomFeatureExtractor(FeatureExtractor):
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

        pacmanX, pacmanY = float(pacmanPos[0]), float(pacmanPos[1])
        width, height = walls.width, walls.height
        foodList = food.asList()
        nearestFoodDistance = min(manhattanDistance(pacmanPos, foodPos) for foodPos in foodList) if foodList else 0.0
        nearestCapsuleDistance = min(manhattanDistance(pacmanPos, capPos) for capPos in capsules) if capsules else 0.0    
        maxDistance = float(width + height)

        features["pacman_x_norm"] = pacmanX / (width - 1.0)
        features["pacman_y_norm"] = pacmanY / (height - 1.0)
        features[f"action={action}"] = 1.0
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