import numpy as np
import util

from sklearn.neural_network import MLPRegressor
from qlearningAgents import PacmanQAgent
from nnFeatureExtractors import MLPRegressorExtractor

class MLPRegressorQAgent(PacmanQAgent):
    def __init__(self, extractor='MLPRegressorExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        self.params = dict(
                hidden_layer_sizes=(64,),
                activation="relu",
                solver="adam",
                learning_rate_init=0.001,
                warm_start=True,
            )
        if "params" in args:
            p = args.pop("params")
            self.params.update(p)            
        PacmanQAgent.__init__(self, **args)
        self.mlp = MLPRegressor(**self.params)
        self.model_initialized = False     

    def assertInitialized(self, X):
        if not self.model_initialized:
            X_dim = X.shape[1]
            print("Initializing MLPRegressor model with input dimension: ", X_dim)
            X0 = np.zeros((1, X_dim))
            y0 = np.zeros(1)
            self.mlp.fit(X0, y0)
            self.model_initialized = True
    
    def getQValue(self, state, action):
        features = self.featExtractor.getFeatures(state, action)
        X = self.featExtractor.toInputVector(features)
        self.assertInitialized(X)
        y_pred = self.mlp.predict(X)
        qValue = float(y_pred[0])
        return qValue

    def update(self, state, action, nextState, reward):
        nextQValue = self.computeValueFromQValues(nextState)
        difference = (reward + self.discount * nextQValue) - self.getQValue(state, action)
        features = self.featExtractor.getFeatures(state, action)
        X = self.featExtractor.toInputVector(features)
        self.assertInitialized(X)
        y = self.getQValue(state, action) + self.alpha * difference
        self.mlp.fit(X.reshape(1, -1), np.array([y]))