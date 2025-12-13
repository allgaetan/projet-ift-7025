import numpy as np
import random
from sklearn.neural_network import MLPRegressor
from qlearningAgents import PacmanQAgent
import util as util
from nnFeatureExtractors import *

class NeuralNetworkQAgent(PacmanQAgent):
    def __init__(self, extractor='NeuralNetworkExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        nn_params = args.pop("nn_params", {})
        PacmanQAgent.__init__(self, **args)

        self.model_initialized = False
        self.batch_size = 1
        self.memory = []
        default = dict(
            hidden_layer_sizes=(64, 64),
            activation="logistic",
            solver="sgd",
            learning_rate_init=0.01,
            warm_start=True,
            max_iter=100
        )
        params = {**default, **nn_params}
        self.mlp = MLPRegressor(**params)

    def _ensure_initialized(self, x_dim):
        if not self.model_initialized:
            X0 = np.zeros((1, x_dim))
            y0 = np.zeros(1)
            self.mlp.fit(X0, y0)
            self.model_initialized = True
    
    def getQValue(self, state, action):
        x = self.featExtractor.getFeatureVector(state, action)
        self._ensure_initialized(x.shape[0])
        predQValue = float(self.mlp.predict(x.reshape(1, -1))[0])
        return predQValue

    def update(self, state, action, nextState, reward):
        x = self.featExtractor.getFeatureVector(state, action)
        target = reward + self.discount * self.computeValueFromQValues(nextState)
        self.memory.append((x, target))
        self._ensure_initialized(x.shape[0])
        Xy = random.sample(self.memory, min(len(self.memory), self.batch_size))
        X = np.array([item[0] for item in Xy])
        y = np.array([item[1] for item in Xy])
        self.mlp.fit(X, y)
    
    def final(self, state):
        PacmanQAgent.final(self, state)
        if self.episodesSoFar == self.numTraining:
            pass